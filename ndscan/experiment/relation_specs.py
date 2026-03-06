import re
from dataclasses import dataclass
from typing import Any

from .fragment import Fragment
from .parameters import ParamHandle, ParamStore
from .relation_expressions import (
    RelationExpressionError,
    compile_safe_relation_expr,
    rewrite_bracket_param_refs,
)
from .utils import path_matches_spec

__all__ = [
    "collect_explicit_param_targets",
    "activate_scan_arg_param_relations",
]


def collect_explicit_param_targets(
    param_stores: dict[str, list[tuple[str, ParamStore]]]
) -> set[tuple[str, str]]:
    explicit = set[tuple[str, str]]()
    for pairs in param_stores.values():
        for _path, store in pairs:
            for handle in store._handles:
                explicit.add((handle.parameter.fqn, handle.owner._stringize_path()))
    return explicit


@dataclass(frozen=True)
class _RelationParamRef:
    fqn: str
    path: str = "*"


def _parse_relation_param_ref(
    spec: Any, context: str, error_type: type[Exception]
) -> _RelationParamRef:
    if isinstance(spec, str):
        fqn = spec
        path = "*"
    elif isinstance(spec, dict):
        try:
            fqn = spec["fqn"]
        except KeyError:
            raise error_type(f"{context} is missing required key 'fqn'")
        path = spec.get("path", "*")
    else:
        raise error_type(f"{context} must be a string FQN or dict with 'fqn'/'path'")

    if not isinstance(fqn, str) or not fqn:
        raise error_type(f"{context}.fqn must be a non-empty string")
    if not isinstance(path, str) or not path:
        raise error_type(f"{context}.path must be a non-empty string")
    return _RelationParamRef(fqn=fqn, path=path)


def _parse_relation_token_ref(
    token: str, context: str, error_type: type[Exception]
) -> _RelationParamRef:
    token = token.strip()
    if not token:
        raise error_type(f"{context} must not be empty")
    if "@" in token:
        fqn, path = token.rsplit("@", 1)
        return _parse_relation_param_ref(
            {"fqn": fqn.strip(), "path": path.strip()}, context, error_type
        )
    return _parse_relation_param_ref(token, context, error_type)


def _suggest_alias(fqn: str, used_aliases: set[str]) -> str:
    base = fqn.rsplit(".", 1)[-1]
    alias = re.sub(r"\W", "_", base)
    if not alias or alias[0].isdigit():
        alias = f"param_{alias}"
    if not alias.isidentifier():
        alias = "param"
    candidate = alias
    i = 1
    while candidate in used_aliases:
        candidate = f"{alias}_{i}"
        i += 1
    return candidate


def _replace_ident(expr: str, old: str, new: str) -> str:
    pattern = r"\b" + re.escape(old) + r"\b"
    return re.sub(pattern, new, expr)


def _build_free_param_handle_indexes(
    fragment: Fragment,
) -> tuple[dict[str, list[ParamHandle]], dict[str, list[ParamHandle]]]:
    attached = dict[str, list[ParamHandle]]()
    fragment._collect_free_param_handles(attached, include_detached=False)
    all_handles = dict[str, list[ParamHandle]]()
    fragment._collect_free_param_handles(all_handles, include_detached=True)
    return attached, all_handles


def _resolve_relation_handle(
    ref: _RelationParamRef,
    attached_index: dict[str, list[ParamHandle]],
    all_handles_index: dict[str, list[ParamHandle]],
    error_type: type[Exception],
) -> ParamHandle:
    all_candidates = all_handles_index.get(ref.fqn, [])
    all_matches = [
        h for h in all_candidates if path_matches_spec(h.owner._fragment_path, ref.path)
    ]
    if not all_matches:
        raise error_type(
            "Relation references parameter with no matching free handle: "
            + f"{ref.fqn}@{ref.path}"
        )

    candidates = attached_index.get(ref.fqn, [])
    matches = [
        h for h in candidates if path_matches_spec(h.owner._fragment_path, ref.path)
    ]
    if not matches:
        detached_paths = ", ".join(
            sorted(h.owner._stringize_path() for h in all_matches)
        )
        raise error_type(
            "Relation references detached subfragment parameter, which is currently "
            "unsupported in scan.relations: "
            + f"{ref.fqn}@{ref.path}. Detached matches: {detached_paths}"
        )
    if len(matches) > 1:
        match_paths = ", ".join(sorted(h.owner._stringize_path() for h in matches))
        raise error_type(
            "Relation reference is ambiguous; specify a narrower path for "
            + f"{ref.fqn}@{ref.path}. Matches: {match_paths}"
        )
    return matches[0]


def _extract_relation_deps_and_expr(
    relation_spec: dict[str, Any], relation_context: str, error_type: type[Exception]
) -> tuple[list[tuple[str, _RelationParamRef]], str]:
    deps_spec = relation_spec.get("deps", [])
    if deps_spec is None:
        deps_spec = []
    if not isinstance(deps_spec, list):
        raise error_type(f"{relation_context}.deps must be a list if specified")

    try:
        expr = relation_spec["expr"]
    except KeyError:
        raise error_type(f"{relation_context} is missing required key 'expr'")
    if not isinstance(expr, str):
        raise error_type(f"{relation_context}.expr must be a string")

    used_aliases = set[str]()
    deps = list[tuple[str, _RelationParamRef]]()
    ref_to_alias = dict[tuple[str, str], str]()

    for i, dep in enumerate(deps_spec):
        dep_context = f"{relation_context}.deps[{i}]"
        alias = None
        if isinstance(dep, dict):
            alias_value = dep.get("alias", None)
            if alias_value is not None:
                if not isinstance(alias_value, str) or not alias_value.isidentifier():
                    raise error_type(
                        f"{dep_context}.alias must be a valid Python identifier"
                    )
                alias = alias_value
        dep_ref = _parse_relation_param_ref(dep, dep_context, error_type)
        if alias is None:
            alias = _suggest_alias(dep_ref.fqn, used_aliases)
        if alias in used_aliases:
            raise error_type(
                f"{dep_context}.alias '{alias}' collides with another dependency alias"
            )
        used_aliases.add(alias)
        key = (dep_ref.fqn, dep_ref.path)
        ref_to_alias[key] = alias
        deps.append((alias, dep_ref))

    rewritten_expr, token_to_temp_alias = rewrite_bracket_param_refs(expr)
    for token, temp_alias in token_to_temp_alias.items():
        ref = _parse_relation_token_ref(token, f"{relation_context}.expr", error_type)
        key = (ref.fqn, ref.path)
        alias = ref_to_alias.get(key)
        if alias is None:
            alias = temp_alias
            if alias in used_aliases:
                alias = _suggest_alias(ref.fqn, used_aliases)
            used_aliases.add(alias)
            ref_to_alias[key] = alias
            deps.append((alias, ref))
        if alias != temp_alias:
            rewritten_expr = _replace_ident(rewritten_expr, temp_alias, alias)

    return deps, rewritten_expr


def _is_relation_active(
    activation_refs: list[_RelationParamRef],
    activation_mode: str,
    explicit_param_targets: set[tuple[str, str]],
) -> bool:
    def ref_matches_explicit(ref: _RelationParamRef) -> bool:
        for fqn, path in explicit_param_targets:
            if fqn != ref.fqn:
                continue
            if path_matches_spec(path.split("/"), ref.path):
                return True
        return False

    if not activation_refs:
        return True
    matches = [ref_matches_explicit(ref) for ref in activation_refs]
    if activation_mode == "all":
        return all(matches)
    return any(matches)


def activate_scan_arg_param_relations(
    fragment: Fragment,
    relation_specs: list[dict[str, Any]],
    explicit_param_targets: set[tuple[str, str]],
    error_type: type[Exception],
) -> None:
    if not relation_specs:
        return
    if not isinstance(relation_specs, list):
        raise error_type("scan.relations must be a list")

    attached_index, all_handles_index = _build_free_param_handle_indexes(fragment)
    for i, relation_spec in enumerate(relation_specs):
        relation_context = f"scan.relations[{i}]"
        if not isinstance(relation_spec, dict):
            raise error_type(f"{relation_context} must be a dictionary")
        if relation_spec.get("enabled", True) is False:
            continue

        if "target" not in relation_spec:
            raise error_type(f"{relation_context} is missing required key 'target'")
        target = _parse_relation_param_ref(
            relation_spec["target"], relation_context + ".target", error_type
        )
        deps, rewritten_expr = _extract_relation_deps_and_expr(
            relation_spec, relation_context, error_type
        )
        dep_aliases = tuple(alias for alias, _dep in deps)
        dep_refs = [dep for _alias, dep in deps]

        activation_mode = relation_spec.get("activation_mode", "any")
        if activation_mode not in ("any", "all"):
            raise error_type(
                f"{relation_context}.activation_mode must be 'any' or 'all'"
            )
        activation_dep_specs = relation_spec.get("activation_deps", None)
        if activation_dep_specs is None:
            activation_refs = dep_refs
        else:
            if not isinstance(activation_dep_specs, list):
                raise error_type(
                    f"{relation_context}.activation_deps must be a list"
                )
            activation_refs = [
                _parse_relation_param_ref(
                    s, f"{relation_context}.activation_deps[{j}]", error_type
                )
                for j, s in enumerate(activation_dep_specs)
            ]

        if not _is_relation_active(
            activation_refs, activation_mode, explicit_param_targets
        ):
            continue
        if any(
            fqn == target.fqn and path_matches_spec(path.split("/"), target.path)
            for fqn, path in explicit_param_targets
        ):
            raise ValueError(
                "Optional relation target is explicitly driven in the same run: "
                + target.fqn
                + "@"
                + target.path
            )

        try:
            fn = compile_safe_relation_expr(rewritten_expr, dep_aliases)
        except RelationExpressionError as e:
            raise error_type(f"{relation_context}: {e}") from None

        target_handle = _resolve_relation_handle(
            target, attached_index, all_handles_index, error_type
        )
        dep_handles = tuple(
            _resolve_relation_handle(dep, attached_index, all_handles_index, error_type)
            for dep in dep_refs
        )
        if any(d is target_handle for d in dep_handles):
            raise error_type(
                f"{relation_context}: target parameter appears in its dependency list"
            )

        fragment._add_runtime_param_relation(target_handle, dep_handles, fn)
