"""dr-copper: copper futures macro analysis pipeline."""

from dr_copper.arima_baseline import (
    acf,
    adf_test,
    fit_arima,
    pacf,
    residuals_diagnostic,
    restore_prices,
)
from dr_copper.backtest import build_backtest_frame, run_backtest, run_backtest_opt
from dr_copper.data_loader import fetch_raw
from dr_copper.features_builder import fetch_features
from dr_copper.garch_model import get_acf, get_pacf, plot_garch, walk_forward_garch
from dr_copper.lstm_model import (
    CopperLSTM,
    CopperLSTMTrainer,
    LSTMConfig,
    LSTMTrainerData,
    build_sequences,
    lstm_signal,
    lstm_test_score,
)
from dr_copper.pca import fit_pca, transform_pca
from dr_copper.regimes import (
    fit_regimes,
    plot_elbow,
    plot_regime_history,
    plot_regime_scatter,
    predict_regimes,
)
from dr_copper.strategy import (
    SignalData,
    SignalSMAStrategy,
    SignalStrategy,
    SMAStrategy,
)
from dr_copper.tearsheet import (
    benchmark_returns_from_prices,
    generate_html_tearsheet,
    print_key_metrics,
)
from dr_copper.viz import (
    plot_explained_variance,
    plot_forecasts,
    plot_loadings_heatmap,
    plot_pc_scores,
    save,
)
from dr_copper.xgb_model import (
    CopperXGBTrainer,
    XGBConfig,
    XGBTrainerData,
    xgb_test_score,
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
    "transform_pca",
    "predict_regimes",
    "build_sequences",
    "CopperLSTM",
    "LSTMConfig",
    "LSTMTrainerData",
    "CopperLSTMTrainer",
    "lstm_test_score",
    "CopperXGBTrainer",
    "XGBConfig",
    "XGBTrainerData",
    "xgb_test_score",
    "lstm_signal",
    "build_backtest_frame",
    "SignalData",
    "SignalStrategy",
    "SMAStrategy",
    "SignalSMAStrategy",
    "benchmark_returns_from_prices",
    "print_key_metrics",
    "run_backtest",
    "generate_html_tearsheet",
    "run_backtest_opt",
]
