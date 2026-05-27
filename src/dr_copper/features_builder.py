import numpy as np
import pandas as pd

PRICE_COLS: list[str] = ["crude", "gold", "copper", "dxy", "sp500"]


def fetch_features(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)

    log_prices = np.log(raw[PRICE_COLS])
    for col in PRICE_COLS:
        out[f"{col}_ret1d"] = log_prices[col].diff(1)  # type: ignore
        out[f"{col}_ret5d"] = log_prices[col].diff(5)  # type: ignore

    out["copper_vol21d"] = log_prices["copper"].diff(1).rolling(21).std() * np.sqrt(252)  # type: ignore

    out[["cli_china", "cli_usa"]] = raw[["cli_china", "cli_usa"]].ffill(axis=0)

    out[["DFII10", "USDCNY"]] = raw[["DFII10", "USDCNY"]]

    out.dropna(inplace=True)

    return out
