"""Adapter layer for the vendored ``sensible_fitting`` library.

This module gives ndscan one stable place to talk to the third-party fitting code.
The rest of ndscan should not need to know how the vendored package is imported or how
its result objects are structured internally.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import importlib
import sys

import numpy as np

_MODEL_FIT_KIND = "model_fit"
_MODEL_FIT_PROVIDER = "sensible_fitting"
_VENDORED_SRC = Path(__file__).resolve().parents[2] / "third_party" / "sensible-fitting" / "src"

_SENSIBLE_MODEL_FACTORIES = {
    "straight_line": "straight_line",
    "gaussian_with_offset": "gaussian_with_offset",
    "gaussian_dip": None,
    "sinusoid": "sinusoid",
    "rabi_oscillation": "rabi_oscillation",
}


@dataclass(frozen=True)
class SensibleModelSpec:
    """Description of one named sensible-fitting model offered by ndscan."""

    model_id: str
    label: str
    parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class ModelFitArtifact:
    """JSON-safe structured fit result shared across runtime and viewer code."""

    model_id: str
    model_name: str
    backend: str
    data_format: str
    parameters: dict[str, dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    plot: dict[str, Any] = field(default_factory=dict)
    kind: str = _MODEL_FIT_KIND
    provider: str = _MODEL_FIT_PROVIDER

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provider": self.provider,
            "model_id": self.model_id,
            "model_name": self.model_name,
            "backend": self.backend,
            "data_format": self.data_format,
            "parameters": dict(self.parameters),
            "stats": dict(self.stats),
            "source": dict(self.source),
            "plot": dict(self.plot),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelFitArtifact":
        return cls(
            model_id=str(data["model_id"]),
            model_name=str(data.get("model_name", data["model_id"])),
            backend=str(data.get("backend", "")),
            data_format=str(data.get("data_format", "normal")),
            parameters={
                str(name): dict(spec)
                for name, spec in dict(data.get("parameters", {})).items()
            },
            stats=dict(data.get("stats", {})),
            source=dict(data.get("source", {})),
            plot=dict(data.get("plot", {})),
            kind=str(data.get("kind", _MODEL_FIT_KIND)),
            provider=str(data.get("provider", _MODEL_FIT_PROVIDER)),
        )


def import_sensible_fitting():
    """Import the vendored ``sensible_fitting`` package.

    ndscan vendors the package under ``third_party/`` so the local source tree can be
    used without a separate environment installation step.
    """

    try:
        return importlib.import_module("sensible_fitting")
    except ModuleNotFoundError as original_error:
        if _VENDORED_SRC.is_dir():
            vendor_src = str(_VENDORED_SRC)
            if vendor_src not in sys.path:
                sys.path.insert(0, vendor_src)
            return importlib.import_module("sensible_fitting")
        raise original_error


def is_sensible_fitting_available() -> bool:
    try:
        import_sensible_fitting()
    except ModuleNotFoundError:
        return False
    return True


def get_builtin_model_specs() -> list[SensibleModelSpec]:
    """Return the built-in sensible-fitting models ndscan exposes by default."""

    sensible = import_sensible_fitting()
    specs = []
    for model_id, factory_name in _SENSIBLE_MODEL_FACTORIES.items():
        if model_id == "gaussian_dip":
            model = _build_gaussian_dip_model()
        else:
            model = getattr(sensible.models, factory_name)()
        specs.append(
            SensibleModelSpec(
                model_id=model_id,
                label=model.name,
                parameter_names=tuple(model.param_names),
            )
        )
    return specs


def build_builtin_model(model_id: str):
    """Construct one built-in sensible-fitting model by ndscan model id."""

    sensible = import_sensible_fitting()
    factory_name = _SENSIBLE_MODEL_FACTORIES.get(model_id, None)
    if factory_name is None:
        if model_id != "gaussian_dip":
            raise ValueError(f"unknown sensible-fitting model id: {model_id}")
        model = _build_gaussian_dip_model()
    else:
        model = getattr(sensible.models, factory_name)()
    if model_id == "straight_line":
        # The vendored straight-line model has no built-in guesser. Give ndscan a
        # simple default seed so the model is immediately usable from viewer fits and
        # basic runtime analyses without emitting warnings about inferred seeds.
        model = model.guess(m=0.0, b=0.0)
    return model


def _build_gaussian_dip_model():
    """Return a Gaussian dip model with offset.

    The vendored library ships a positive-amplitude Gaussian-with-offset model. For
    spectroscopy-style survival probabilities, ndscan often wants the inverted shape:

    ``y = y0 - a * exp(-0.5 * ((x - x0) / sigma) ** 2)``

    where ``y0`` is the off-resonant baseline and ``a`` is the dip depth.
    """

    sensible = import_sensible_fitting()

    def gaussian_dip_func(x, x0, y0, a, sigma):
        return y0 - a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    base = (
        sensible.Model.from_function(gaussian_dip_func, name="gaussian dip")
        .bound(a=(0.0, None), sigma=(1e-12, None))
    )

    def init_gaussian_dip(x, y, guess_state):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.size == 0 or y_arr.size == 0:
            return

        if guess_state.is_unset("y0"):
            guess_state.y0 = float(np.max(y_arr))
        if guess_state.is_unset("a"):
            guess_state.a = max(0.0, float(np.max(y_arr) - np.min(y_arr)))
        if guess_state.is_unset("x0"):
            guess_state.x0 = float(x_arr[np.argmin(y_arr)])
        if guess_state.is_unset("sigma"):
            span = float(np.max(x_arr) - np.min(x_arr))
            guess_state.sigma = 0.2 * span if span > 0.0 else 1.0

    return base.with_guesser(init_gaussian_dip).derive(
        "fwhm",
        lambda params: 2.35482 * params["sigma"],
        doc="Full-width at half minimum (2.35482 * sigma)",
    )


def fit_data_with_model(
    model,
    fit_data,
    *,
    model_id: str,
    backend: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> tuple[Any, ModelFitArtifact]:
    """Fit ``fit_data`` with one sensible-fitting model and return run + artifact."""

    fit_kwargs = {"backend": _default_backend_for_fit_data(fit_data, backend)}
    if backend_options is not None:
        fit_kwargs["backend_options"] = dict(backend_options)
    run = model.fit(fit_data, **fit_kwargs).squeeze()
    artifact = build_model_fit_artifact(run, model_id=model_id, source=source)
    return run, artifact


def fit_data_with_builtin_model(
    model_id: str,
    fit_data,
    *,
    backend: str | None = None,
    backend_options: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
):
    """Fit ``fit_data`` with one named built-in model and return run + artifact."""

    model = build_builtin_model(model_id)
    return fit_data_with_model(
        model,
        fit_data,
        model_id=model_id,
        backend=backend,
        backend_options=backend_options,
        source=source,
    )


def build_model_fit_artifact(
    run,
    *,
    model_id: str,
    source: Mapping[str, Any] | None = None,
) -> ModelFitArtifact:
    """Convert one scalar sensible-fitting ``Run`` into a serializable artifact."""

    squeezed = run.squeeze()
    results = squeezed.results
    parameters = dict[str, dict[str, Any]]()
    for name, param in results.params.items():
        parameters[name] = {
            "value": _json_safe(param.value),
            "stderr": _json_safe(param.stderr),
            "fixed": bool(param.fixed),
            "derived": bool(param.derived),
        }

    plot = {}
    if squeezed.data is not None:
        meta = dict(squeezed.data.get("meta", {}))
        for key in ("x_label", "y_label", "label"):
            if key in meta:
                plot[key] = _json_safe(meta[key])

    stats = _json_safe_mapping(results.stats)
    stats["summary"] = results.summary()
    stats["success"] = _json_safe(squeezed.success)
    stats["message"] = _json_safe(squeezed.message)

    return ModelFitArtifact(
        model_id=model_id,
        model_name=str(getattr(squeezed.model, "name", model_id)),
        backend=str(squeezed.backend),
        data_format=str(squeezed.data_format),
        parameters=parameters,
        stats=stats,
        source=_json_safe_mapping(dict(source or {})),
        plot=plot,
    )


def coerce_model_fit_artifact(artifact: ModelFitArtifact | Mapping[str, Any]) -> ModelFitArtifact:
    if isinstance(artifact, ModelFitArtifact):
        return artifact
    return ModelFitArtifact.from_dict(artifact)


def evaluate_model_fit_artifact(
    artifact: ModelFitArtifact | Mapping[str, Any],
    x: Any,
) -> np.ndarray:
    """Evaluate the fitted model at ``x`` using the artifact's fitted parameters."""

    coerced = coerce_model_fit_artifact(artifact)
    model = build_builtin_model(coerced.model_id)
    params = {
        name: spec["value"]
        for name, spec in coerced.parameters.items()
        if "value" in spec
    }
    return np.asarray(model.eval(np.asarray(x, dtype=float), params=params), dtype=float)


def curve_points_for_artifact(
    artifact: ModelFitArtifact | Mapping[str, Any],
    *,
    x_values: Sequence[float] | np.ndarray | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
    num_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sampled curve points for one artifact.

    The artifact remains the canonical fit object; sampling is done on demand for
    plotting helpers that still want explicit arrays.
    """

    if x_values is None:
        if x_min is None or x_max is None:
            raise ValueError("curve_points_for_artifact requires x_values or x_min/x_max")
        x_values = np.linspace(float(x_min), float(x_max), int(num_points))
    x_array = np.asarray(x_values, dtype=float)
    return x_array, evaluate_model_fit_artifact(artifact, x_array)


def artifact_parameter_value(
    artifact: ModelFitArtifact | Mapping[str, Any], name: str
) -> Any:
    """Return one fitted parameter value from a model-fit artifact."""

    coerced = coerce_model_fit_artifact(artifact)
    return coerced.parameters[name]["value"]


def artifact_summary(artifact: ModelFitArtifact | Mapping[str, Any]) -> str:
    """Return a compact human-readable parameter summary for one fit artifact."""

    import_sensible_fitting()
    format_uncertainty = importlib.import_module(
        "sensible_fitting.viz"
    ).uncertainty_to_string
    coerced = coerce_model_fit_artifact(artifact)
    parts = []
    for name, spec in coerced.parameters.items():
        value = spec.get("value", None)
        stderr = spec.get("stderr", None)
        if value is None:
            continue
        if stderr is None:
            parts.append(f"{name}={_scalar_string(value)}")
        else:
            try:
                parts.append(
                    f"{name}={format_uncertainty(float(value), float(stderr), precision='auto')}"
                )
            except Exception:
                parts.append(f"{name}={_scalar_string(value)} ± {_scalar_string(stderr)}")
    return ", ".join(parts)


def _scalar_string(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def _json_safe_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in mapping.items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _default_backend_for_fit_data(fit_data: Any, backend: str | None) -> str:
    if backend is not None:
        return backend
    data_format = getattr(fit_data, "data_format", None)
    if data_format == "normal":
        return "scipy.curve_fit"
    return "scipy.minimize"
