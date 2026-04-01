"""Viewer-side 1D fitting helpers for the prepared-runtime plotter.

These fits are intentionally client-side only. They are useful for fast exploratory
inspection in the live viewer, but they are not part of the experiment runtime or the
persisted ndscan schema.

The fitting interface is backend-neutral so the viewer can:

- use a tiny built-in fallback backend,
- use the vendored ``sensible_fitting`` models when available,
- and still hand the rest of the viewer one consistent ``FitResult`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ...fits.sensible import (
    ModelFitArtifact,
    artifact_summary,
    curve_points_for_artifact,
    fit_data_with_builtin_model,
    get_builtin_model_specs,
    import_sensible_fitting,
    is_sensible_fitting_available,
)


@dataclass(frozen=True)
class FitModel:
    """Description of one viewer-side fit model."""

    model_id: str
    label: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class FitRequest:
    """Input data for a viewer-side fit."""

    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class FitResult:
    """Result returned by a viewer-side fit backend."""

    model_id: str
    curve_x: np.ndarray
    curve_y: np.ndarray
    parameters: dict[str, float]
    artifact: ModelFitArtifact | None = None

    def summary(self) -> str:
        """Return a compact parameter summary for small UI labels."""
        if self.artifact is not None:
            return artifact_summary(self.artifact)
        return ", ".join(f"{name}={value:.6g}" for name, value in self.parameters.items())


class FitBackend(Protocol):
    """Protocol implemented by viewer-side fit backends."""

    def models(self) -> list[FitModel]:
        """Return the models offered by this backend."""

    def fit(self, model_id: str, request: FitRequest) -> FitResult:
        """Fit one model to one 1D data slice."""


class BuiltinFitBackend:
    """Small built-in fit backend used until a richer lab fit library is plugged in."""

    _MODELS = (
        FitModel("linear", "linear", ("slope", "offset")),
        FitModel("quadratic", "quadratic", ("a", "b", "c")),
    )

    def models(self) -> list[FitModel]:
        return list(self._MODELS)

    def fit(self, model_id: str, request: FitRequest) -> FitResult:
        x = np.asarray(request.x, dtype=float)
        y = np.asarray(request.y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("viewer-side fits expect 1D point arrays")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        if model_id == "linear":
            if len(x) < 2:
                raise ValueError("linear fit needs at least 2 points")
            slope, offset = np.polyfit(x, y, deg=1)
            curve_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            curve_y = slope * curve_x + offset
            return FitResult(
                model_id=model_id,
                curve_x=curve_x,
                curve_y=curve_y,
                parameters={"slope": float(slope), "offset": float(offset)},
            )

        if model_id == "quadratic":
            if len(x) < 3:
                raise ValueError("quadratic fit needs at least 3 points")
            a, b, c = np.polyfit(x, y, deg=2)
            curve_x = np.linspace(float(np.min(x)), float(np.max(x)), 200)
            curve_y = a * curve_x**2 + b * curve_x + c
            return FitResult(
                model_id=model_id,
                curve_x=curve_x,
                curve_y=curve_y,
                parameters={"a": float(a), "b": float(b), "c": float(c)},
            )

        raise ValueError(f"unknown viewer-side fit model: {model_id}")


class SensibleFitBackend:
    """Viewer fit backend backed by the vendored ``sensible_fitting`` models."""

    def __init__(self):
        import_sensible_fitting()
        self._models = [
            FitModel(spec.model_id, spec.label, spec.parameter_names)
            for spec in get_builtin_model_specs()
        ]

    def models(self) -> list[FitModel]:
        return list(self._models)

    def fit(self, model_id: str, request: FitRequest) -> FitResult:
        sensible = import_sensible_fitting()
        x = np.asarray(request.x, dtype=float)
        y = np.asarray(request.y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("viewer-side fits expect 1D point arrays")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        data = sensible.FitData.normal(
            x=x,
            y=y,
            x_label="x",
            y_label="y",
            label="viewer",
        )
        _, artifact = fit_data_with_builtin_model(model_id, data)
        curve_x, curve_y = curve_points_for_artifact(
            artifact,
            x_min=float(np.min(x)),
            x_max=float(np.max(x)),
            num_points=200,
        )
        parameters = {
            name: float(spec["value"])
            for name, spec in artifact.parameters.items()
            if spec.get("value", None) is not None
        }
        return FitResult(
            model_id=model_id,
            curve_x=curve_x,
            curve_y=curve_y,
            parameters=parameters,
            artifact=artifact,
        )


class CombinedFitBackend:
    """Combine multiple fit backends behind one model-id namespace."""

    def __init__(self, *backends: FitBackend):
        self._backends = tuple(backends)
        self._model_to_backend = dict[str, FitBackend]()
        self._models = []
        for backend in self._backends:
            for model in backend.models():
                if model.model_id in self._model_to_backend:
                    raise ValueError(f"duplicate fit model id: {model.model_id}")
                self._model_to_backend[model.model_id] = backend
                self._models.append(model)

    def models(self) -> list[FitModel]:
        return list(self._models)

    def fit(self, model_id: str, request: FitRequest) -> FitResult:
        backend = self._model_to_backend.get(model_id, None)
        if backend is None:
            raise ValueError(f"unknown viewer-side fit model: {model_id}")
        return backend.fit(model_id, request)


def default_fit_backend() -> FitBackend:
    """Return the default viewer fit backend for this ndscan checkout."""

    builtin = BuiltinFitBackend()
    if is_sensible_fitting_available():
        return CombinedFitBackend(SensibleFitBackend(), builtin)
    return builtin
