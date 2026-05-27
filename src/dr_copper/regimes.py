from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from dr_copper.viz import save

DEFAULT_PCS = ["PC1", "PC2", "PC3"]

REGIME_COLORS = ["#b87333", "#7eb8c9", "#9ecf9e", "#cf9e9e", "#c9b87e", "#b87eb8"]


def fit_regimes(
    scores: pd.DataFrame,
    n_regimes: int = 4,
    random_state: int = 42,
) -> tuple[pd.Series, KMeans, pd.DataFrame]:
    X = scores[DEFAULT_PCS].values

    km = KMeans(n_clusters=n_regimes, random_state=random_state, n_init="auto")
    raw_labels = km.fit_predict(X)

    # --- Canonical ordering: sort regimes by PC1 centroid (low → high) ---
    centroid_pc1 = km.cluster_centers_[:, 0]
    order = np.argsort(centroid_pc1)  # original label → rank
    remap = {orig: new for new, orig in enumerate(order)}
    ordered_labels = np.vectorize(remap.get)(raw_labels)

    labels = pd.Series(ordered_labels, index=scores.index, name="regime")

    # Summary table
    df = scores[DEFAULT_PCS].copy()
    df["regime"] = labels
    summary = (
        df.groupby("regime")[DEFAULT_PCS]
        .mean()
        .round(3)
        .assign(n_obs=df.groupby("regime").size())
    )

    return labels, km, summary


def plot_elbow(
    scores: pd.DataFrame,
    save_path: Path,
) -> None:
    max_k = 8
    X = scores[DEFAULT_PCS].values
    inertias = []
    ks = range(2, max_k + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#161616")
    ax.plot(list(ks), inertias, color="#b87333", marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Number of regimes (k)", labelpad=8)
    ax.set_ylabel("Inertia", labelpad=8)
    ax.set_title("Elbow Curve — K-Means on PC Scores", color="#e8a060", fontsize=13)
    ax.grid(color="#2a2a2a", linestyle="--", linewidth=0.5)
    ax.tick_params(colors="#e0e0e0")
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # type: ignore

    save(fig, save_path)


def plot_regime_history(
    labels: pd.Series,
    copper_price: pd.Series,
    save_path: Path,
) -> None:
    n_regimes = labels.nunique()
    colors = REGIME_COLORS[:n_regimes]

    fig, ax = plt.subplots(figsize=(13, 4), facecolor="#0d0d0d")
    ax.set_facecolor("#161616")

    # Shade regime bands
    prev_date = labels.index[0]
    prev_label = labels.iloc[0]

    for date, label in labels.items():
        if label != prev_label:
            ax.axvspan(
                prev_date,  # type: ignore
                date,  # type: ignore
                color=colors[prev_label],
                alpha=0.25,
                linewidth=0,  # type: ignore
            )
            prev_date = date
            prev_label = label
    ax.axvspan(
        prev_date,  # type: ignore
        labels.index[-1],
        color=colors[prev_label],
        alpha=0.25,
        linewidth=0,  # type: ignore
    )

    # Regime label as step line
    ax.step(labels.index, labels.values, color="#555555", linewidth=0.6, where="post")  # type: ignore
    ax.set_ylabel("Regime", color="#e0e0e0")
    ax.set_yticks(range(n_regimes))
    ax.set_yticklabels([f"R{i}" for i in range(n_regimes)], color="#e0e0e0")

    # Optional price overlay
    if copper_price is not None:
        ax2 = ax.twinx()
        cp = copper_price.reindex(labels.index).ffill()
        ax2.plot(cp.index, cp, color="#b87333", linewidth=1.2, alpha=0.85)
        ax2.set_ylabel("HG=F (USD)", color="#b87333", fontsize=9)
        ax2.tick_params(colors="#b87333", labelsize=8)

    legend_patches = [
        mpatches.Patch(color=colors[i], alpha=0.6, label=f"Regime {i}")
        for i in range(n_regimes)
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper left",
        framealpha=0.2,
        edgecolor="#5a5a5a",
        labelcolor="#e0e0e0",
        fontsize=8,
    )

    ax.set_title(
        "Macro Regimes Over Time (K-Means on PC Scores)",
        color="#e8a060",
        fontsize=13,
        pad=10,
    )
    ax.set_xlabel("Date", labelpad=8)
    ax.grid(color="#2a2a2a", linestyle="--", linewidth=0.4)
    ax.tick_params(colors="#e0e0e0")

    save(fig, save_path)


def plot_regime_scatter(
    scores: pd.DataFrame,
    labels: pd.Series,
    save_path: Path,
    x_pc: str = "PC1",
    y_pc: str = "PC2",
    z_pc: str = "PC3",
) -> None:
    n_regimes = labels.nunique()
    colors = REGIME_COLORS[:n_regimes]

    fig = plt.figure(figsize=(10, 7), facecolor="#0d0d0d")

    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#161616")
    # Darken pane backgrounds
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):  # type: ignore
        pane.fill = True
        pane.set_facecolor("#111111")
        pane.set_edgecolor("#2a2a2a")

    for regime in range(n_regimes):
        mask = labels == regime
        xs = scores.loc[mask, x_pc]
        ys = scores.loc[mask, y_pc]

        zs = scores.loc[mask, z_pc]
        ax.scatter(
            xs,
            ys,
            zs,  # type: ignore
            color=colors[regime],
            alpha=0.30,
            s=10,
            label=f"Regime {regime}",
            depthshade=True,
        )
        # cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
        # ax.scatter(
        #     cx,
        #     cy,
        #     cz,  # type: ignore
        #     color=colors[regime],
        #     s=140,
        #     marker="X",
        #     edgecolors="white",
        #     linewidths=0.8,
        #     zorder=5,
        #     depthshade=False,
        # )
        # ax.text(cx, cy, cz, f"  R{regime}", color=colors[regime], fontsize=9)

    # Axes labels & style
    ax.set_xlabel(x_pc, labelpad=8, color="#e0e0e0")
    ax.set_ylabel(y_pc, labelpad=8, color="#e0e0e0")
    ax.tick_params(colors="#e0e0e0", labelsize=8)

    ax.set_zlabel(z_pc, labelpad=-30, color="#e0e0e0")  # type: ignore[attr-defined]
    ax.tick_params(axis="z", colors="#e0e0e0", labelsize=8)  # type: ignore
    title = f"Regime Scatter — {x_pc} / {y_pc} / {z_pc}"

    ax.set_title(title, color="#e8a060", fontsize=13, pad=12)
    ax.legend(framealpha=0.2, edgecolor="#5a5a5a", labelcolor="#e0e0e0", fontsize=8)

    save(fig, save_path)
