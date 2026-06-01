import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def _adf_test_results(series: pd.Series, alpha: float = 0.05) -> dict:
    stat, pval, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")  # type: ignore
    return {
        "statistic": stat,
        "p_value": pval,
        "lags_used": lags,
        "n_obs": nobs,
        "critical_values": crit,
        "is_stationary": pval < alpha,
    }


def adf_test(series: pd.Series, alpha: float = 0.05) -> None:
    results = _adf_test_results(series, alpha)
    print("ADF Statistic: ", results["statistic"])
    print("P-Value: ", results["p_value"])
    print("Critical Values:")
    for thres, adf_stat in results["critical_values"].items():
        print("\t%s: %.2f" % (thres, adf_stat))
    print("Is stationary: ", results["is_stationary"])


def fit_arima(train: pd.DataFrame, d: int = 0) -> ARIMA:
    return pm.auto_arima(
        train,
        d=d,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        max_order=None,  # type: ignore
        information_criterion="aicc",
        error_action="ignore",
    )


def restore_prices(prices, log_returns):
    p0 = prices.iloc[0]

    restored_prices = p0 * np.exp(log_returns.cumsum())

    restored_prices = pd.concat(
        [pd.Series([p0], index=[prices.index[0]]), restored_prices]
    )

    return restored_prices


def acf(train):
    plt.rc("figure", figsize=(7, 4))
    plot_acf(train)
    plt.xlabel("Lags", fontsize=18)
    plt.ylabel("Correlation", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Autocorrelation Plot", fontsize=20)
    plt.tight_layout()
    plt.show()


def pacf(train):
    plt.rc("figure", figsize=(7, 4))
    plot_pacf(train, method="ywm")
    plt.xlabel("Lags", fontsize=18)
    plt.ylabel("Correlation", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("Partial Autocorrelation Plot", fontsize=20)
    plt.tight_layout()
    plt.show()


def residuals_diagnostic(model: ARIMA, alpha: float = 0.05, auto_arima: bool = False):
    if auto_arima:
        resid = model.resid()  # type: ignore
        order = model.order
    else:
        resid = model.resid  # type: ignore
        order = model.specification["order"]  # type: ignore

    # Ljung-Box | checking autocorrelation
    lb = acorr_ljungbox(resid, lags=10, return_df=True)
    lb_pass = (lb["lb_pvalue"] > alpha).all()

    # Jarque-Bera | checking normal distribution
    jb_stat, jb_pval, jb_skew, jb_kurt = jarque_bera(resid)

    # Engle's Test for Autoregressive Conditional Heteroscedasticity (ARCH)
    lm_stat, lm_pval, f_stat, f_pval = het_arch(resid, nlags=12)  # type: ignore

    results = {
        "ljung_box": {
            "table": lb,
            "all_pass": lb_pass,
            "interpretation": (
                "No serial correlation in residuals"
                if lb_pass
                else "⚠ Serial correlation detected — consider higher p/q"
            ),
        },
        "jarque_bera": {
            "statistic": jb_stat,
            "p_value": jb_pval,
            "skewness": jb_skew,
            "kurtosis": jb_kurt,
            "is_normal": jb_pval > alpha,
            "interpretation": (
                "Residuals approximately normal"
                if jb_pval > alpha
                else "⚠ Non-normal residuals — GARCH captures fat tails"
            ),
        },
        "het_arch": {
            "statistics_lm": lm_stat,
            "p_value_lm": lm_pval,
            "statistics_f": f_stat,
            "p_value_f": f_pval,
            "interpretation": (
                "No strong evidence of ARCH effects."
                if lm_pval > alpha
                else "⚠ ARCH effects present → GARCH justified"
            ),
        },
    }

    print("=" * 60)
    print(f"ARIMA{order} Residual Diagnostics")
    print("=" * 60)
    print(f"\nLjung-Box: {results['ljung_box']['interpretation']}")
    print(f"Jarque-Bera: p={jb_pval:.4f}  {results['jarque_bera']['interpretation']}")
    print(f"Engle's ARCH LM test:  {results['het_arch']['interpretation']}")
    print()
