from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

COPPER_PALETTE = {
    "bg": "#0d0d0d",
    "panel": "#161616",
    "copper": "#b87333",
    "accent": "#e8a060",
    "muted": "#5a5a5a",
    "text": "#e0e0e0",
    "grid": "#2a2a2a",
}

plt.rcParams.update(
    {
        "figure.facecolor": COPPER_PALETTE["bg"],
        "axes.facecolor": COPPER_PALETTE["panel"],
        "axes.edgecolor": COPPER_PALETTE["muted"],
        "axes.labelcolor": COPPER_PALETTE["text"],
        "axes.titlecolor": COPPER_PALETTE["text"],
        "xtick.color": COPPER_PALETTE["text"],
        "ytick.color": COPPER_PALETTE["text"],
        "text.color": COPPER_PALETTE["text"],
        "grid.color": COPPER_PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "font.family": "monospace",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    }
)


def save(fig, save_path):
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_loadings_heatmap(
    loadings: pd.DataFrame,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COPPER_PALETTE["bg"])

    # Diverging copper-toned colormap
    cmap = sns.diverging_palette(h_neg=30, h_pos=210, s=90, l=40, as_cmap=True)

    sns.heatmap(
        loadings,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8, "color": COPPER_PALETTE["bg"]},
        linewidths=0.4,
        linecolor=COPPER_PALETTE["bg"],
        cbar_kws={"shrink": 0.75, "label": "Loading"},
    )

    ax.set_title(
        "PCA Loadings — Feature × Component",
        pad=14,
        color=COPPER_PALETTE["accent"],
        fontsize=13,
    )
    ax.set_xlabel("Principal Component", labelpad=8)
    ax.set_ylabel("Feature", labelpad=8)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(COPPER_PALETTE["text"])  # type: ignore
    cbar.ax.tick_params(colors=COPPER_PALETTE["text"])  # type: ignore

    save(fig, save_path)


def plot_explained_variance(
    evr: np.ndarray,
    save_path: Path,
) -> None:
    n = len(evr)
    cumulative = np.cumsum(evr)
    labels = [f"PC{i + 1}" for i in range(n)]

    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor=COPPER_PALETTE["bg"])
    ax2 = ax1.twinx()

    # Bars — individual
    bars = ax1.bar(
        labels,
        evr * 100,
        color=COPPER_PALETTE["copper"],
        alpha=0.85,
        zorder=3,
        width=0.55,
    )

    # Annotate bars
    for bar, val in zip(bars, evr * 100):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            color=COPPER_PALETTE["accent"],
            fontsize=9,
        )

    # Line — cumulative
    ax2.plot(
        labels,
        cumulative * 100,
        color=COPPER_PALETTE["accent"],
        marker="o",
        markersize=6,
        linewidth=2,
        zorder=4,
        label="Cumulative",
    )
    ax2.axhline(60, color=COPPER_PALETTE["muted"], linestyle=":", linewidth=1)
    ax2.text(n - 0.5, 81, "80%", color=COPPER_PALETTE["muted"], fontsize=8, va="bottom")

    # Axes
    ax1.set_ylim(0, max(evr * 100) * 1.35)
    ax2.set_ylim(0, 110)
    ax1.set_xlabel("Component", labelpad=8)
    ax1.set_ylabel("Individual EVR (%)", labelpad=8)
    ax2.set_ylabel("Cumulative EVR (%)", labelpad=8)
    ax1.set_title(
        "Explained Variance by Principal Component",
        pad=12,
        color=COPPER_PALETTE["accent"],
        fontsize=13,
    )

    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    ax1.grid(axis="y", zorder=0)
    ax2.tick_params(colors=COPPER_PALETTE["text"])

    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.88, 0.85),
        framealpha=0.2,
        edgecolor=COPPER_PALETTE["muted"],
        labelcolor=COPPER_PALETTE["text"],
    )

    save(fig, save_path)


def plot_pc_scores(
    scores: pd.DataFrame,
    copper_price: pd.Series,
    save_path: Path,
) -> None:
    n_pcs = scores.shape[1]
    figsize = (13, 2.5 * n_pcs)

    fig, axes = plt.subplots(
        n_pcs,
        1,
        figsize=figsize,
        facecolor=COPPER_PALETTE["bg"],
        sharex=True,
    )
    if n_pcs == 1:
        axes = [axes]

    colors = [
        COPPER_PALETTE["copper"],
        COPPER_PALETTE["accent"],
        "#7eb8c9",
        "#9ecf9e",
        "#cf9e9e",
    ]

    for i, (ax, pc) in enumerate(zip(axes, scores.columns)):
        color = colors[i % len(colors)]
        s = scores[pc]

        ax.fill_between(
            s.index, s, 0, where=s >= 0, color=color, alpha=0.35, linewidth=0
        )
        ax.fill_between(
            s.index, s, 0, where=s < 0, color=color, alpha=0.15, linewidth=0
        )
        ax.plot(s.index, s, color=color, linewidth=0.8, alpha=0.9)
        ax.axhline(0, color=COPPER_PALETTE["muted"], linewidth=0.7)

        ax.set_ylabel(pc, labelpad=6, color=color, fontsize=10)
        ax.grid(axis="x")
        ax.grid(axis="y")
        ax.tick_params(axis="both", labelsize=8)

        # Overlay copper price on PC1 (right axis)
        if i == 0:
            ax2 = ax.twinx()
            cp = copper_price.reindex(s.index).ffill()
            ax2.plot(
                cp.index, cp, color="#ffffff", linewidth=0.9, alpha=0.45, linestyle="--"
            )
            ax2.set_ylabel("HG=F (USD)", color="#aaaaaa", fontsize=8)
            ax2.tick_params(colors="#aaaaaa", labelsize=7)

    axes[0].set_title(
        "Principal Component Scores",
        pad=10,
        color=COPPER_PALETTE["accent"],
        fontsize=13,
    )
    axes[-1].set_xlabel("Date", labelpad=8)

    fig.subplots_adjust(hspace=0.12)
    save(fig, save_path)


def plot_forecasts(train, test, forecast: list[float], title, grid_flg=True):
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Train", color=COPPER_PALETTE["copper"])
    plt.plot(test.index, test, label="Test", color=COPPER_PALETTE["accent"])
    plt.plot(test.index, forecast, label="Forecast")

    plt.title(title)
    plt.xlabel("Copper Price")
    plt.ylabel("Date")
    plt.grid(grid_flg)
    plt.legend()

    plt.show()
