"""Boolean condition helpers for per-group ROI occupancy statistics.

The intended model is deliberately simple:

- ``group`` is the repetition axis, i.e. one independent copy of the experiment,
- conditions describe one logical group at a time,
- statistics are reported both pooled across groups and separately for each group.

Cross-group quantification is intentionally not modelled here. If a future experiment
really needs it, it can build a more specialised analysis layer at that point.

Conditions can be built:

- structurally, from :class:`Occupied`, :class:`Empty`, :class:`And`, :class:`Or`,
  :class:`Not`,
- from a sparse DNF clause representation ``[(image, roi, state), ...]``,
- or from a compact GUI-friendly syntax such as ``"1[0,1] & 2[!3]"``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np

from examples._binomial_chunk_analysis import estimate_probability_from_counts

__all__ = [
    "Always",
    "And",
    "ClauseTerm",
    "ConditionalBinomialResult",
    "Empty",
    "Not",
    "Occupied",
    "Or",
    "all_of",
    "any_of",
    "condition_from_clauses",
    "conditional_binomial",
    "counts_to_occupancy_stack",
    "parse_condition_syntax",
]

OccupancyByImage = tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Always:
    """Condition that always evaluates to ``True``."""


@dataclass(frozen=True)
class Occupied:
    """One occupied trap at ``(image, roi)`` within one logical group."""

    image_index: int
    roi_index: int


@dataclass(frozen=True)
class Empty:
    """One empty trap at ``(image, roi)`` within one logical group."""

    image_index: int
    roi_index: int


@dataclass(frozen=True)
class Not:
    condition: object


@dataclass(frozen=True)
class And:
    conditions: tuple[object, ...]


@dataclass(frozen=True)
class Or:
    conditions: tuple[object, ...]


ClauseTerm = tuple[int, int, int]

_TOKEN_RE = re.compile(
    r"""
    \s*
    (?:
        (?P<LPAREN>\()
        | (?P<RPAREN>\))
        | (?P<AND>&)
        | (?P<OR>\|)
        | (?P<NOT>!)
        | (?P<ATOM>(?:i)?\d+\[[^\[\]]*\])
    )
    """,
    re.VERBOSE,
)
_ATOM_RE = re.compile(r"(?P<image>(?:i)?\d+)\[(?P<terms>[^\[\]]*)\]$")
_ROI_TERM_RE = re.compile(r"(?P<neg>!)?(?P<roi>\d+)$")


def all_of(*conditions) -> And:
    """Return the conjunction of ``conditions``."""

    return And(tuple(conditions))


def any_of(*conditions) -> Or:
    """Return the disjunction of ``conditions``."""

    return Or(tuple(conditions))


def _term_to_condition(image_index: int, roi_index: int, state: int):
    if int(state) == 1:
        return Occupied(int(image_index), int(roi_index))
    return Empty(int(image_index), int(roi_index))


def condition_from_clauses(clauses: list[list[ClauseTerm]] | tuple[tuple[ClauseTerm, ...], ...]):
    """Return a condition from sparse DNF clauses.

    ``clauses`` is interpreted as:

    - outer list: OR
    - inner list: AND
    - tuple: ``(image_index, roi_index, state)``, where ``state`` is ``0`` or ``1``
    """

    if not clauses:
        return Always()

    disjuncts = []
    for clause in clauses:
        if not clause:
            disjuncts.append(Always())
            continue
        terms = tuple(_term_to_condition(*term) for term in clause)
        if len(terms) == 1:
            disjuncts.append(terms[0])
        else:
            disjuncts.append(And(terms))

    if len(disjuncts) == 1:
        return disjuncts[0]
    return Or(tuple(disjuncts))


@dataclass(frozen=True)
class _SyntaxToken:
    kind: str
    value: str
    position: int


def _tokenize_condition_syntax(text: str) -> list[_SyntaxToken]:
    tokens: list[_SyntaxToken] = []
    position = 0
    while position < len(text):
        match = _TOKEN_RE.match(text, position)
        if match is None:
            if text[position:].strip() == "":
                break
            raise ValueError(
                f"Invalid ROI condition syntax near {text[position:position + 16]!r}"
            )
        position = match.end()
        kind = match.lastgroup
        if kind is None:
            continue
        tokens.append(_SyntaxToken(kind, match.group(kind), match.start(kind)))
    return tokens


class _ConditionSyntaxParser:
    def __init__(self, tokens: list[_SyntaxToken]):
        self._tokens = tokens
        self._index = 0

    def parse(self):
        if not self._tokens:
            return Always()
        condition = self._parse_or()
        if self._peek() is not None:
            token = self._peek()
            raise ValueError(
                f"Unexpected token {token.value!r} at position {token.position}"
            )
        return condition

    def _peek(self) -> _SyntaxToken | None:
        if self._index >= len(self._tokens):
            return None
        return self._tokens[self._index]

    def _accept(self, kind: str) -> _SyntaxToken | None:
        token = self._peek()
        if token is None or token.kind != kind:
            return None
        self._index += 1
        return token

    def _expect(self, kind: str) -> _SyntaxToken:
        token = self._accept(kind)
        if token is None:
            next_token = self._peek()
            if next_token is None:
                raise ValueError(f"Expected {kind} but reached end of expression")
            raise ValueError(
                f"Expected {kind} at position {next_token.position}, got {next_token.value!r}"
            )
        return token

    def _parse_or(self):
        conditions = [self._parse_and()]
        while self._accept("OR") is not None:
            conditions.append(self._parse_and())
        if len(conditions) == 1:
            return conditions[0]
        return Or(tuple(conditions))

    def _parse_and(self):
        conditions = [self._parse_unary()]
        while self._accept("AND") is not None:
            conditions.append(self._parse_unary())
        if len(conditions) == 1:
            return conditions[0]
        return And(tuple(conditions))

    def _parse_unary(self):
        if self._accept("NOT") is not None:
            return Not(self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        if self._accept("LPAREN") is not None:
            condition = self._parse_or()
            self._expect("RPAREN")
            return condition
        atom = self._expect("ATOM")
        return _condition_from_atom_text(atom.value)


def _condition_from_atom_text(text: str):
    match = _ATOM_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"Invalid ROI image block {text!r}")

    image_text = match.group("image")
    image_index = int(image_text[1:] if image_text.startswith("i") else image_text)
    raw_terms = [term.strip() for term in match.group("terms").split(",")]
    if not raw_terms or any(term == "" for term in raw_terms):
        raise ValueError(
            f"Image block {text!r} must contain one or more ROI selectors like '0' or '!3'"
        )

    conditions = []
    for raw_term in raw_terms:
        roi_match = _ROI_TERM_RE.fullmatch(raw_term)
        if roi_match is None:
            raise ValueError(
                f"Invalid ROI selector {raw_term!r} inside image block {text!r}"
            )
        roi_index = int(roi_match.group("roi"))
        if roi_match.group("neg") is None:
            conditions.append(Occupied(image_index, roi_index))
        else:
            conditions.append(Empty(image_index, roi_index))

    if len(conditions) == 1:
        return conditions[0]
    return And(tuple(conditions))


def parse_condition_syntax(text: str):
    """Parse one compact boolean condition expression.

    The syntax uses normal boolean operators and one compact per-image atom:

    - ``&`` for AND
    - ``|`` for OR
    - ``!`` for NOT
    - ``(``, ``)`` for grouping
    - ``<image>[<roi selectors>]`` for one image block

    Inside one image block, comma-separated ROI selectors are ANDed together:

    - ``1[0,1]`` means image 1 ROI 0 bright AND ROI 1 bright
    - ``2[!3]`` means image 2 ROI 3 dark

    Example::

        (1[0,1] & 2[!3]) | 1[4,5]
    """

    stripped = text.strip()
    if not stripped:
        return Always()
    return _ConditionSyntaxParser(_tokenize_condition_syntax(stripped)).parse()


@dataclass(frozen=True)
class ConditionalBinomialResult:
    """Conditional probability estimates pooled and per group."""

    num_selected_by_group: np.ndarray
    num_successes_by_group: np.ndarray
    probability_by_group: np.ndarray
    probability_error_by_group: np.ndarray
    pooled_num_selected: int
    pooled_num_successes: int
    pooled_probability: float
    pooled_probability_error: float


def counts_to_occupancy_stack(
    counts_by_image: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    threshold: int,
) -> OccupancyByImage:
    """Return one boolean occupancy array per image.

    Each returned array has shape ``(shots, group, roi)``. All images must share the
    same number of shots and groups, but the ROI dimension may differ between images.
    """

    arrays = [np.asarray(counts) for counts in counts_by_image]
    if not arrays:
        raise ValueError("At least one image counts array is required")
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("Each counts array must have shape (shots, group, roi)")
    first_shots, first_groups, _ = arrays[0].shape
    for array in arrays[1:]:
        shots, groups, _ = array.shape
        if shots != first_shots or groups != first_groups:
            raise ValueError(
                "All image counts arrays must share the same number of shots and groups"
            )
    return tuple(np.asarray(array >= int(threshold), dtype=bool) for array in arrays)


def _normalise_occupancy_by_image(
    occupancy: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
) -> OccupancyByImage:
    if isinstance(occupancy, np.ndarray):
        occupancy_array = np.asarray(occupancy, dtype=bool)
        if occupancy_array.ndim != 4:
            raise ValueError("occupancy must have shape (shots, image, group, roi)")
        return tuple(
            np.asarray(occupancy_array[:, image_index, :, :], dtype=bool)
            for image_index in range(occupancy_array.shape[1])
        )

    arrays = tuple(np.asarray(image, dtype=bool) for image in occupancy)
    if not arrays:
        raise ValueError("At least one occupancy image array is required")
    if any(array.ndim != 3 for array in arrays):
        raise ValueError("Each occupancy image must have shape (shots, group, roi)")
    first_shots, first_groups, _ = arrays[0].shape
    for array in arrays[1:]:
        shots, groups, _ = array.shape
        if shots != first_shots or groups != first_groups:
            raise ValueError(
                "All occupancy images must share the same number of shots and groups"
            )
    return arrays


def _evaluate_condition_for_group(
    occupancy_by_image: OccupancyByImage,
    condition: Any,
    group_index: int,
) -> np.ndarray:
    if isinstance(condition, Always):
        return np.ones(occupancy_by_image[0].shape[0], dtype=bool)
    if isinstance(condition, Occupied):
        return occupancy_by_image[int(condition.image_index)][
            :,
            int(group_index),
            int(condition.roi_index),
        ]
    if isinstance(condition, Empty):
        return ~occupancy_by_image[int(condition.image_index)][
            :,
            int(group_index),
            int(condition.roi_index),
        ]
    if isinstance(condition, Not):
        return ~_evaluate_condition_for_group(
            occupancy_by_image, condition.condition, group_index
        )
    if isinstance(condition, And):
        if not condition.conditions:
            return np.ones(occupancy_by_image[0].shape[0], dtype=bool)
        result = np.ones(occupancy_by_image[0].shape[0], dtype=bool)
        for child in condition.conditions:
            result &= _evaluate_condition_for_group(
                occupancy_by_image, child, group_index
            )
        return result
    if isinstance(condition, Or):
        if not condition.conditions:
            return np.zeros(occupancy_by_image[0].shape[0], dtype=bool)
        result = np.zeros(occupancy_by_image[0].shape[0], dtype=bool)
        for child in condition.conditions:
            result |= _evaluate_condition_for_group(
                occupancy_by_image, child, group_index
            )
        return result
    raise TypeError(f"Unsupported condition type: {type(condition)!r}")


def conditional_binomial(
    occupancy: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    event,
    given=None,
) -> ConditionalBinomialResult:
    """Return conditional binomial statistics pooled and by group.

    ``occupancy`` may be either:

    - one ``(shots, image, group, roi)`` boolean array, or
    - one sequence of ``(shots, group, roi)`` boolean arrays, one per image.
    """

    occupancy_by_image = _normalise_occupancy_by_image(occupancy)

    given_condition = Always() if given is None else given
    num_groups = occupancy_by_image[0].shape[1]

    num_selected_by_group = np.empty(num_groups, dtype=int)
    num_successes_by_group = np.empty(num_groups, dtype=int)
    probability_by_group = np.empty(num_groups, dtype=float)
    probability_error_by_group = np.empty(num_groups, dtype=float)

    for group_index in range(num_groups):
        selected = _evaluate_condition_for_group(
            occupancy_by_image, given_condition, group_index
        )
        successes = selected & _evaluate_condition_for_group(
            occupancy_by_image, event, group_index
        )
        num_selected = int(np.sum(selected))
        num_successes = int(np.sum(successes))
        probability, probability_error = estimate_probability_from_counts(
            num_successes, num_selected
        )

        num_selected_by_group[group_index] = num_selected
        num_successes_by_group[group_index] = num_successes
        probability_by_group[group_index] = probability
        probability_error_by_group[group_index] = probability_error

    pooled_num_selected = int(np.sum(num_selected_by_group))
    pooled_num_successes = int(np.sum(num_successes_by_group))
    pooled_probability, pooled_probability_error = estimate_probability_from_counts(
        pooled_num_successes,
        pooled_num_selected,
    )

    return ConditionalBinomialResult(
        num_selected_by_group=num_selected_by_group,
        num_successes_by_group=num_successes_by_group,
        probability_by_group=probability_by_group,
        probability_error_by_group=probability_error_by_group,
        pooled_num_selected=pooled_num_selected,
        pooled_num_successes=pooled_num_successes,
        pooled_probability=pooled_probability,
        pooled_probability_error=pooled_probability_error,
    )
