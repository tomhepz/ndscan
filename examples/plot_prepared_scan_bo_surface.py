"""Plot the synthetic BO surface used by prepared_scan_bayesian_optimisation.py.

Usage:

    python examples/plot_prepared_scan_bo_surface.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    background = 0.12 * (x**2 + y**2)
    well_a = -1.6 * np.exp(-((x + 1.1) ** 2 / 0.5 + (y - 0.8) ** 2 / 0.7))
    well_b = -1.2 * np.exp(-((x - 1.4) ** 2 / 0.6 + (y + 1.0) ** 2 / 0.5))
    ripple = 0.08 * np.sin(2.5 * x) * np.cos(2.0 * y)
    return background + well_a + well_b + ripple


def main() -> None:
    x = np.linspace(-2.5, 2.5, 220)
    y = np.linspace(-2.5, 2.5, 220)
    xx, yy = np.meshgrid(x, y)
    zz = surface(xx, yy)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        xx,
        yy,
        zz,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        rcount=180,
        ccount=180,
        alpha=0.95,
    )
    ax.contour(
        xx,
        yy,
        zz,
        levels=18,
        zdir="z",
        offset=np.min(zz) - 0.15,
        cmap="viridis",
        linewidths=0.8,
    )

    ax.set_title("Prepared Runtime BO Example Surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("cost")
    ax.set_zlim(np.min(zz) - 0.15, np.max(zz))
    ax.view_init(elev=28, azim=-135)

    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08, label="cost")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
