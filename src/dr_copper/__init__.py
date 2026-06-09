"""dr-copper: copper futures macro analysis pipeline."""

from dr_copper.arima_baseline import (
    acf,
    adf_test,
    fit_arima,
    pacf,
    residuals_diagnostic,
    restore_prices,
)
from dr_copper.data_loader import fetch_raw
from dr_copper.features_builder import fetch_features
from dr_copper.garch_model import get_acf, get_pacf, plot_garch, walk_forward_garch
from dr_copper.pca import fit_pca
from dr_copper.regimes import (
    fit_regimes,
    plot_elbow,
    plot_regime_history,
    plot_regime_scatter,
)
from dr_copper.viz import (
    plot_explained_variance,
    plot_forecasts,
    plot_loadings_heatmap,
    plot_pc_scores,
    save,
)

__all__ = [
    "fetch_raw",
    "fetch_features",
    "fit_pca",
    "plot_loadings_heatmap",
    "plot_explained_variance",
    "plot_pc_scores",
    "save",
    "fit_regimes",
    "plot_elbow",
    "plot_regime_history",
    "plot_regime_scatter",
    "adf_test",
    "fit_arima",
    "restore_prices",
    "plot_forecasts",
    "acf",
    "pacf",
    "residuals_diagnostic",
    "get_acf",
    "get_pacf",
    "walk_forward_garch",
    "plot_garch",
]
