from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.stattools import acf, pacf

from dr_copper.viz import COPPER_PALETTE, save


def walk_forward_garch(
    log_returns: pd.Series,
    min_train_size: int = 252,  # 1 trading year minimum
    refit_every: int = 21,  # refit monthly
    mean: str = "Constant",
    ar_lags: int = 1,
    p: int = 1,
    q: int = 1,
    dist: str = "studentst",
    verbose: bool = True,
) -> tuple[pd.DataFrame, object]:
    n = len(log_returns)
    idx = log_returns.index
    scaled = log_returns * 100

    # Output containers
    cond_vol = np.full(n, np.nan)
    omegas = np.full(n, np.nan)
    alphas = np.full(n, np.nan)
    betas = np.full(n, np.nan)

    current_model = None

    for t in range(min_train_size, n):
        refit_due = (t - min_train_size) % refit_every == 0

        if refit_due or current_model is None:
            train = scaled.iloc[:t]

            am = arch_model(
                train,
                mean=mean,  # type: ignore
                lags=ar_lags if mean == "AR" else 0,
                vol="GARCH",
                p=p,
                q=q,
                dist=dist,  # type: ignore
                rescale=False,
            )
            res = am.fit(disp="off", show_warning=False)
            current_model = res

            if verbose and refit_due and (t - min_train_size) % (refit_every * 10) == 0:
                print(
                    f"  Walk-forward: t={t}/{n}  ({100 * t / n:.0f}%)  train_size={t}"
                )

        if current_model is None:
            continue

        # One-step-ahead forecast from position t (forecasting t+1 is position t)
        # We use the last in-sample conditional variance as h_t, then
        # compute h_{t+1} = ω + α·ε_t² + β·h_t  (analytical 1-step)
        params = current_model.params
        omega_ = params.get("omega", np.nan)
        alpha_ = params.get("alpha[1]", np.nan)
        beta_ = params.get("beta[1]", np.nan)

        # Last in-sample values
        h_last = current_model.conditional_volatility.iloc[-1] ** 2  # type: ignore
        e_last = current_model.resid.iloc[-1]  # type: ignore

        h_next = omega_ + alpha_ * (e_last**2) + beta_ * h_last
        vol_next = np.sqrt(max(h_next, 0.0)) / 100.0  # back to log-return scale

        cond_vol[t] = vol_next
        omegas[t] = omega_
        alphas[t] = alpha_
        betas[t] = beta_

    result = pd.DataFrame(
        {
            "garch_cond_vol": cond_vol,
            "garch_cond_var": cond_vol**2,
            "garch_vol_ann": cond_vol * np.sqrt(252),
            "omega": omegas,
            "alpha": alphas,
            "beta": betas,
        },
        index=idx,
    )
    result["persistence"] = result["alpha"] + result["beta"]

    if verbose:
        n_valid = result["garch_cond_vol"].notna().sum()
        print(f"\nWalk-forward complete.  Valid observations: {n_valid}/{n}")

    return result, current_model


def get_acf(returns, nlags=250, aplha=0.05):
    acf_vals, acf_ci = acf(returns, nlags=250, alpha=0.05)  # type: ignore

    acf_df = pd.DataFrame(
        {
            "lag": range(len(acf_vals)),
            "acf": acf_vals,
            "ci_lower": acf_ci[:, 0] - acf_vals,
            "ci_upper": acf_ci[:, 1] - acf_vals,
        }
    )
    acf_df["significant"] = (acf_df["acf"] < acf_df["ci_lower"]) | (
        acf_df["acf"] > acf_df["ci_upper"]
    )

    return acf_df


def get_pacf(returns, nlags=250, aplha=0.05):
    pacf_vals, pacf_ci = pacf(returns, nlags=250, alpha=0.05, method="ywm")

    pacf_df = pd.DataFrame(
        {
            "lag": range(len(pacf_vals)),
            "pacf": pacf_vals,
            "ci_lower": pacf_ci[:, 0] - pacf_vals,
            "ci_upper": pacf_ci[:, 1] - pacf_vals,
        }
    )
    pacf_df["significant"] = (pacf_df["pacf"] < pacf_df["ci_lower"]) | (
        pacf_df["pacf"] > pacf_df["ci_upper"]
    )

    return pacf_df


def plot_garch(price, log_ret, garch_wf, split_date, save_path: Path):
    figsize = (13, 5)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        facecolor=COPPER_PALETTE["bg"],
        sharex=True,
    )

    axes[0].plot(
        garch_wf.index,
        garch_wf["garch_vol_ann"],
        lw=0.8,
        color="#2E86AB",
        label="Walk-forward (no look-ahead)",
        alpha=0.9,
    )
    # Train/test split line
    axes[0].axvline(
        split_date,
        color="red",
        lw=1.2,
        ls=":",
        label=f"Train/Test split ({split_date.date()})",
    )
    axes[0].set_title("GARCH(1,1) Conditional Vol — Walk-forward")
    axes[0].legend()

    ax2 = axes[0].twinx()
    ax2.plot(
        log_ret.index,
        log_ret,
        color=COPPER_PALETTE["copper"],
        linewidth=0.9,
        alpha=0.45,
    )
    ax2.set_ylabel("Log returns", color="#aaaaaa", fontsize=8)
    ax2.tick_params(colors="#aaaaaa", labelsize=7)

    axes[1].plot(
        price.index,
        price,
        color=COPPER_PALETTE["copper"],
        label="Copper price",
        linewidth=0.9,
        alpha=0.45,
    )
    axes[1].set_ylabel("HG=F (USD)", color="#aaaaaa", fontsize=8)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    save(fig, save_path)
