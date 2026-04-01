"""Viewer-side 1D fitting helpers for the prepared-runtime plotter.

These fits are intentionally client-side only. They are useful for fast exploratory
inspection in the live viewer, but they are not part of the experiment runtime or the
persisted ndscan schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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

    def summary(self) -> str:
        """Return a compact parameter summary for small UI labels."""
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
