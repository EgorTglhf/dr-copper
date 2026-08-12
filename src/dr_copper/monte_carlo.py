import numpy as np
import QuantLib as ql


def gbm(n_paths, n_steps, mu, sigma_hat, dt, S0):
    np.random.seed(42)
    Z = np.random.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma_hat**2) * dt + sigma_hat * np.sqrt(dt) * Z
    log_cum = np.cumsum(log_increments, axis=1)

    gbm_paths = S0 * np.exp(
        np.hstack([np.zeros((n_paths, 1)), log_cum])
    )  # (n_paths, n_steps+1)

    return gbm_paths


def heston(n_paths, n_steps, v0, kappa, theta, xi, rho, r_rate, q_rate, S0):
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r_rate, day_count))
    dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q_rate, day_count))

    heston_process = ql.HestonProcess(
        risk_free_ts, dividend_ts, spot_handle, v0, kappa, theta, xi, rho
    )

    time_grid = ql.TimeGrid(n_steps / 252, n_steps)
    n_factors = heston_process.factors()  # 2: price, variance
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(
            n_factors * n_steps, ql.UniformRandomGenerator(seed=42)
        )
    )
    path_generator = ql.GaussianMultiPathGenerator(
        heston_process, list(time_grid), rng, False
    )

    heston_paths = np.zeros((n_paths, n_steps + 1))
    heston_var_paths = np.zeros((n_paths, n_steps + 1))
    for i in range(n_paths):
        sample = path_generator.next()
        multi_path = sample.value()
        heston_paths[i, :] = np.array(multi_path[0])  # price factor
        heston_var_paths[i, :] = np.array(multi_path[1])  # variance factor

    return heston_paths
