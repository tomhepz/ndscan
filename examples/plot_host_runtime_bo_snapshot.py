"""Plot a NUBO-style GP corner plot from a host-runtime HDF5 snapshot.

Usage:

    python examples/plot_host_runtime_bo_snapshot.py <snapshot.h5> [--site path/to/site]

This script is intentionally offline-only:

- it reads the final or preview HDF5 snapshot,
- reconstructs the end-of-run GP from saved observations,
- and renders a corner-plot style view similar to the standalone NUBO scripts.

The plotter expects a host-runtime site whose ``scan.point_policy`` metadata looks
like:

- ``kind = "ask_tell_optimiser"``,
- ``backend.kind = "nubo_bayesian_optimisation"``,
- ``observation_extractor.kind = "scalar_channel"``.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from ndscan.results.scan_site_reader import HostRuntimeSiteData, read_host_runtime_snapshot

try:
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
    from nubo.models import GaussianProcess, fit_gp
    from nubo.optimisation import single
except ModuleNotFoundError as exc:
    raise ImportError(
        "plot_host_runtime_bo_snapshot requires the optional Bayesian optimisation "
        "stack (torch, gpytorch, nubo)"
    ) from exc


SOURCE_COLORS = {
    "seed": "#7a7a7a",
    "bo": "#005f73",
    "explore": "#bb3e03",
    "observed": "#444444",
}


def _site_path_from_argument(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.split("/") if part)


def _sorted_keys(keys: Sequence[str]) -> list[str]:
    def key_index(name: str) -> int:
        try:
            return int(name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 10**9

    return sorted(keys, key=key_index)


def _pseudoparam_label(site: HostRuntimeSiteData, key: str) -> str:
    schema = site.pseudoparams[key]["variable"]
    return schema.get("description") or schema.get("name") or key


def _parameter_label(site: HostRuntimeSiteData, key: str) -> str:
    schema = site.parameters[key]["param"]
    return schema.get("description") or schema["fqn"].split(".")[-1]


def _choose_bo_input_keys(
    site: HostRuntimeSiteData,
    dims: int,
) -> tuple[str, list[str], list[str]]:
    pseudoparam_keys = _sorted_keys(site.pseudoparams.keys())
    if len(pseudoparam_keys) == dims:
        return (
            "pseudoparam",
            pseudoparam_keys,
            [_pseudoparam_label(site, key) for key in pseudoparam_keys],
        )

    scanned_parameter_keys = [
        key
        for key in _sorted_keys(site.parameters.keys())
        if site.parameters[key].get("is_scanned", False)
    ]
    if len(scanned_parameter_keys) == dims:
        return (
            "param",
            scanned_parameter_keys,
            [_parameter_label(site, key) for key in scanned_parameter_keys],
        )

    parameter_keys = _sorted_keys(site.parameters.keys())
    if len(parameter_keys) == dims:
        return (
            "param",
            parameter_keys,
            [_parameter_label(site, key) for key in parameter_keys],
        )

    raise ValueError(
        f"Could not infer {dims} optimiser input dimensions from site data "
        f"(pseudoparams={len(pseudoparam_keys)}, scanned_params={len(scanned_parameter_keys)}, "
        f"params={len(parameter_keys)})"
    )


def _fit_exact_gp_model(
    x_train: torch.Tensor,
    y_train_score: torch.Tensor,
    y_train_err: torch.Tensor,
    *,
    lr: float,
    steps: int,
) -> tuple[GaussianProcess, FixedNoiseGaussianLikelihood]:
    likelihood = FixedNoiseGaussianLikelihood(
        noise=y_train_err.square(),
        learn_additional_noise=True,
    )
    gp = GaussianProcess(x_train, y_train_score, likelihood=likelihood)
    fit_gp(
        x_train,
        y_train_score,
        gp=gp,
        likelihood=likelihood,
        lr=lr,
        steps=steps,
    )
    gp.eval()
    likelihood.eval()
    return gp, likelihood


def _posterior_mean_std(
    gp: GaussianProcess,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 1:
        x = x.reshape(1, -1)
    gp.eval()
    with torch.no_grad(), torch.amp.autocast("cpu", enabled=False):
        posterior = gp(x)
    mean = posterior.mean
    std = posterior.variance.clamp_min(1e-12).sqrt()
    return mean, std


def _surrogate_argmax(
    gp: GaussianProcess,
    bounds: torch.Tensor,
    *,
    num_starts: int,
) -> torch.Tensor:
    train_inputs = gp.train_inputs[0]

    def negative_gp_mean(x: torch.Tensor | np.ndarray) -> float:
        if isinstance(x, np.ndarray):
            xt = torch.as_tensor(x, dtype=train_inputs.dtype, device=train_inputs.device)
        else:
            xt = x.to(dtype=train_inputs.dtype, device=train_inputs.device)
        xt = xt.reshape(1, -1)
        mean, _ = _posterior_mean_std(gp, xt)
        return -float(mean.item())

    x_max, _ = single(
        func=negative_gp_mean,
        bounds=bounds,
        method="L-BFGS-B",
        num_starts=num_starts,
    )
    return x_max.reshape(-1)


def _decode_bo_site(site: HostRuntimeSiteData) -> dict[str, object]:
    point_policy = site.metadata.get("scan.point_policy")
    if not isinstance(point_policy, dict):
        raise ValueError("Site does not contain scan.point_policy metadata")
    if point_policy.get("kind") != "ask_tell_optimiser":
        raise ValueError("Site does not use an ask/tell optimiser point policy")

    backend = point_policy.get("backend", {})
    if backend.get("kind") != "nubo_bayesian_optimisation":
        raise ValueError("Site does not use the NUBO Bayesian optimisation backend")

    extractor = point_policy.get("observation_extractor", {})
    if extractor.get("kind") != "scalar_channel":
        raise ValueError(
            "This plotter currently expects a scalar-channel objective extractor"
        )

    dims = int(backend["dims"])
    x_kind, x_keys, x_labels = _choose_bo_input_keys(site, dims)
    x_obs = np.column_stack([np.asarray(site.point_data[key], dtype=float) for key in x_keys])

    channel_key = extractor["channel_key"]
    objective = np.asarray(site.point_data[channel_key], dtype=float)
    minimise = bool(backend.get("minimise", True))
    objective_score = -objective if minimise else objective

    noise_channel_key = extractor.get("noise_channel_key")
    if noise_channel_key is None:
        floor = float(backend.get("observation_noise_floor", 1e-6))
        objective_err = np.full_like(objective_score, floor, dtype=float)
    else:
        objective_err = np.asarray(site.point_data[noise_channel_key], dtype=float)

    decision_source = np.asarray(
        site.point_data.get("metadata.decision_source", ["observed"] * len(objective_score))
    )

    return {
        "backend": backend,
        "extractor": extractor,
        "x_kind": x_kind,
        "x_keys": x_keys,
        "x_labels": x_labels,
        "x_obs": x_obs,
        "objective": objective,
        "objective_score": objective_score,
        "objective_err": objective_err,
        "decision_source": decision_source,
    }


def _plot_gp_corner(
    *,
    site: HostRuntimeSiteData,
    gp: GaussianProcess,
    bounds: torch.Tensor,
    x_obs: np.ndarray,
    objective: np.ndarray,
    objective_err: np.ndarray,
    decision_source: np.ndarray,
    x_labels: Sequence[str],
    minimise: bool,
    argmax_num_starts: int,
    n_points: int = 60,
    figsize: float = 3.0,
) -> tuple[plt.Figure, np.ndarray]:
    dim = x_obs.shape[1]
    center_x = _surrogate_argmax(
        gp,
        bounds,
        num_starts=argmax_num_starts,
    )
    center_score, _ = _posterior_mean_std(gp, center_x)
    center_objective = -float(center_score.item()) if minimise else float(center_score.item())

    mins, maxs = bounds[0].cpu().numpy(), bounds[1].cpu().numpy()
    ranges = list(zip(mins.tolist(), maxs.tolist()))

    def score_to_objective(values: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return -values if minimise else values

    mean_fields = []
    std_fields = []
    for i, j in itertools.combinations(range(dim), 2):
        xi = np.linspace(*ranges[i], n_points)
        xj = np.linspace(*ranges[j], n_points)
        Xi, Xj = np.meshgrid(xi, xj)
        x_flat = np.tile(center_x.cpu().numpy(), (Xi.size, 1))
        x_flat[:, i] = Xi.ravel()
        x_flat[:, j] = Xj.ravel()
        grid = torch.as_tensor(x_flat, dtype=bounds.dtype, device=bounds.device)
        mu_score, std = _posterior_mean_std(gp, grid)
        mean_fields.append(score_to_objective(mu_score))
        std_fields.append(std.detach().cpu().numpy())

    mean_vmin = (
        np.concatenate(mean_fields).min() if mean_fields else float(np.min(objective))
    )
    mean_vmax = (
        np.concatenate(mean_fields).max() if mean_fields else float(np.max(objective))
    )
    std_vmax = np.concatenate(std_fields).max() if std_fields else float(np.max(objective_err))

    fig, axes = plt.subplots(
        dim,
        dim,
        figsize=(dim * figsize, dim * figsize),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    legend_handles = [
        Line2D([0], [0], color="b", lw=1.5, label="GP mean objective"),
        Line2D([0], [0], color="b", lw=6, alpha=0.2, label="GP +/- 1 sigma"),
    ]
    for source in ["seed", "bo", "explore", "observed"]:
        if source not in decision_source:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=5,
                markerfacecolor="none",
                markeredgecolor=SOURCE_COLORS[source],
                linestyle="none",
                label=source,
            )
        )

    for i, j in itertools.combinations(range(dim), 2):
        xi = np.linspace(*ranges[i], n_points)
        xj = np.linspace(*ranges[j], n_points)
        Xi, Xj = np.meshgrid(xi, xj)
        x_flat = np.tile(center_x.cpu().numpy(), (Xi.size, 1))
        x_flat[:, i] = Xi.ravel()
        x_flat[:, j] = Xj.ravel()
        grid = torch.as_tensor(x_flat, dtype=bounds.dtype, device=bounds.device)
        mu_score, std = _posterior_mean_std(gp, grid)
        mu = score_to_objective(mu_score).reshape(n_points, n_points)
        sigma = std.detach().cpu().numpy().reshape(n_points, n_points)

        ax_mean = axes[j, i]
        ax_mean.contourf(
            Xi,
            Xj,
            mu,
            levels=30,
            cmap="plasma",
            vmin=mean_vmin,
            vmax=mean_vmax,
        )
        ax_std = axes[i, j]
        ax_std.contourf(
            Xi,
            Xj,
            sigma,
            levels=30,
            cmap="cividis",
            vmin=0.0,
            vmax=std_vmax,
        )
        for source in np.unique(decision_source):
            mask = decision_source == source
            color = SOURCE_COLORS.get(str(source), SOURCE_COLORS["observed"])
            ax_mean.scatter(
                x_obs[mask, i],
                x_obs[mask, j],
                s=12,
                facecolors="none",
                edgecolors=color,
                linewidths=0.7,
                alpha=0.7,
            )
            ax_std.scatter(
                x_obs[mask, i],
                x_obs[mask, j],
                s=12,
                facecolors="none",
                edgecolors=color,
                linewidths=0.7,
                alpha=0.7,
            )
        ax_mean.plot(center_x[i].item(), center_x[j].item(), "rx")
        ax_std.plot(center_x[i].item(), center_x[j].item(), "rx")
        ax_mean.set_xlabel(x_labels[i])
        ax_mean.set_ylabel(x_labels[j])
        ax_std.set_xlabel(x_labels[i])
        ax_std.set_ylabel(x_labels[j])

    for i in range(dim):
        xi = np.linspace(*ranges[i], n_points)
        xline = np.tile(center_x.cpu().numpy(), (n_points, 1))
        xline[:, i] = xi
        xline_tensor = torch.as_tensor(xline, dtype=bounds.dtype, device=bounds.device)
        mu_score, std = _posterior_mean_std(gp, xline_tensor)
        mu = score_to_objective(mu_score)
        sigma = std.detach().cpu().numpy()

        ax = axes[i, i]
        ax.plot(xi, mu, "b-", label="GP mean objective")
        ax.fill_between(xi, mu - sigma, mu + sigma, color="b", alpha=0.2)
        for source in np.unique(decision_source):
            mask = decision_source == source
            color = SOURCE_COLORS.get(str(source), SOURCE_COLORS["observed"])
            ax.errorbar(
                x_obs[mask, i],
                objective[mask],
                yerr=objective_err[mask],
                fmt="o",
                mfc="none",
                mec=color,
                ecolor=color,
                capsize=0,
                alpha=0.8,
            )
        ax.plot(center_x[i].item(), center_objective, "rx")
        ax.set_xlabel(x_labels[i])
        ax.set_ylabel("Objective")
        if i == 0:
            ax.legend(handles=legend_handles, loc="best")

    title = "/".join(site.path) if site.path else "root"
    fig.suptitle(f"{title}: GP refit from final snapshot")
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot")
    parser.add_argument(
        "--site",
        default="",
        help="Slash-separated site path, e.g. optimiser/subscan. Defaults to the root site.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=100,
        help="Grid resolution per plotted axis. Defaults to 100 for a denser offline refit.",
    )
    parser.add_argument(
        "--fit-steps",
        type=int,
        default=None,
        help="Override the offline GP refit optimisation steps.",
    )
    parser.add_argument(
        "--fit-lr",
        type=float,
        default=None,
        help="Override the offline GP refit learning rate.",
    )
    parser.add_argument(
        "--argmax-num-starts",
        type=int,
        default=None,
        help="Override the number of optimisation restarts used to locate the plotted surrogate optimum.",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=None,
        help="Override the minimum observation noise floor used in the offline GP refit.",
    )
    args = parser.parse_args()

    snapshot = read_host_runtime_snapshot(args.snapshot)
    site = snapshot.get_site(_site_path_from_argument(args.site))
    bo_site = _decode_bo_site(site)

    backend = bo_site["backend"]
    fit_steps = int(args.fit_steps if args.fit_steps is not None else max(int(backend.get("fit_steps", 400)), 800))
    fit_lr = float(args.fit_lr if args.fit_lr is not None else backend.get("fit_lr", 0.05))
    noise_floor = float(
        args.noise_floor
        if args.noise_floor is not None
        else backend.get("observation_noise_floor", 1e-6)
    )
    argmax_num_starts = int(
        args.argmax_num_starts
        if args.argmax_num_starts is not None
        else max(int(backend.get("surrogate_num_starts", 20)), 24)
    )
    bounds = torch.as_tensor(backend["bounds"], dtype=torch.float64)
    x_obs = torch.as_tensor(bo_site["x_obs"], dtype=torch.float64)
    objective_score = torch.as_tensor(bo_site["objective_score"], dtype=torch.float64)
    objective_err = torch.as_tensor(bo_site["objective_err"], dtype=torch.float64)

    gp, _ = _fit_exact_gp_model(
        x_obs,
        objective_score,
        objective_err.clamp_min(noise_floor),
        lr=fit_lr,
        steps=fit_steps,
    )

    _plot_gp_corner(
        site=site,
        gp=gp,
        bounds=bounds,
        x_obs=np.asarray(bo_site["x_obs"], dtype=float),
        objective=np.asarray(bo_site["objective"], dtype=float),
        objective_err=np.asarray(bo_site["objective_err"], dtype=float),
        decision_source=np.asarray(bo_site["decision_source"]),
        x_labels=bo_site["x_labels"],
        minimise=bool(backend.get("minimise", True)),
        argmax_num_starts=argmax_num_starts,
        n_points=args.grid_points,
    )
    plt.show()


if __name__ == "__main__":
    main()
