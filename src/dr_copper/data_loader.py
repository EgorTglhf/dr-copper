import pandas as pd
from openbb import obb

FUTURES_TICKERS: dict[str, str] = {
    "copper": "HG=F",
    "gold": "GC=F",
    "crude": "CL=F",
}

INDEX_TICKERS: dict[str, str] = {
    "sp500": "^SPX",
    "dxy": "DX-Y.NYB",
}

CLI_COUNTRIES: dict[str, str] = {
    "cli_china": "china",
    "cli_usa": "united_states",
}

CURRENCY: dict[str, str] = {"USDCNY": "USDCNY"}

FRED_SERIES: dict[str, str] = {
    "real_yield_10y": "DFII10",  # Daily real yield (%)
}


def _fetch_futures(
    tickers: dict[str, str],
    start: str,
) -> pd.DataFrame:
    raw_futures = obb.derivatives.futures.historical(  # type: ignore
        symbol=list(tickers.values()), start_date=start
    ).to_df()[["close", "symbol"]]

    raw_futures = raw_futures.pivot(columns="symbol", values="close")

    inv = {v: k for k, v in tickers.items()}
    raw_futures = raw_futures.rename(columns=inv)

    return raw_futures


def _fetch_indexes(
    tickers: dict[str, str],
    start: str,
) -> pd.DataFrame:
    raw_index = obb.index.price.historical(  # type: ignore
        symbol=list(tickers.values()), start_date=start
    ).to_df()[["close", "symbol"]]

    raw_index = raw_index.pivot(columns="symbol", values="close")

    inv = {v: k for k, v in tickers.items()}
    raw_index = raw_index.rename(columns=inv)

    return raw_index


def _fetch_fred(
    tickers: dict[str, str],
    start: str,
) -> pd.DataFrame:
    raw_fred = obb.economy.fred_series(  # type: ignore
        symbol=list(tickers.values()), start_date=start
    ).to_df()

    return raw_fred


def _fetch_cli(
    tickers: dict[str, str],
    start: str,
) -> pd.DataFrame:
    raw_cli = obb.economy.composite_leading_indicator(  # type: ignore
        country=",".join(tickers.values()), start_date=start
    ).to_df()

    raw_cli = raw_cli.pivot(columns="country", values="value")

    inv = {
        "China": "cli_china",
        "United States": "cli_usa",
    }
    raw_cli = raw_cli.rename(columns=inv)

    return raw_cli


def _fetch_currency(
    tickers: dict[str, str],
    start: str,
) -> pd.DataFrame:
    raw_cur = obb.currency.price.historical(  # type: ignore
        symbol=list(tickers.values()), start_date=start
    ).to_df()[["close"]]

    raw_cur = raw_cur.rename(columns={"close": "USDCNY"})

    return raw_cur


def fetch_raw(
    start: str = "2010-01-01",
) -> pd.DataFrame:
    raw_futures = _fetch_futures(FUTURES_TICKERS, start)
    raw_index = _fetch_indexes(INDEX_TICKERS, start)
    raw_fred = _fetch_fred(FRED_SERIES, start)
    raw_cli = _fetch_cli(CLI_COUNTRIES, start)
    raw_cur = _fetch_currency(CURRENCY, start)

    raw = (
        raw_futures.join(raw_index, how="left")
        .join(raw_fred, how="left")
        .join(raw_cli, how="left")
        .join(raw_cur, how="left")
    )

    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()

    return raw
