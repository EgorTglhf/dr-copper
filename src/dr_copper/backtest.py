import backtrader as bt
import pandas as pd

from .strategy import (
    SignalData,
    SignalStrategy,
)


def build_backtest_frame(
    signal: pd.DataFrame, prices: pd.DataFrame, lag_days: int = 1
) -> pd.DataFrame:
    aligned_signal = signal.reindex(prices.index, fill_value=0.0)
    lagged_signal = aligned_signal.shift(lag_days)

    frame = prices.copy()
    frame["signal"] = lagged_signal

    frame.index = frame.index.astype("datetime64[ns]")

    return frame


def run_backtest_opt(
    frame: pd.DataFrame,
    strategy: bt.Strategy = SignalStrategy,
    initial_cash: float = 100_000.0,
    commission: float = 2.5,
    margin: int = 10000,
    mult: int = 25000,
    commtype: int = bt.CommInfoBase.COMM_FIXED,
    slippage: float = 0.0015,  # 3 ticks of 0.0005 each
    target_percent: list[float] = [0.1, 0.5, 0.9],
    printlog: bool = False,
):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=commission, margin=margin, mult=mult, commtype=commtype
    )
    cerebro.broker.set_slippage_fixed(fixed=slippage)

    data = SignalData(dataname=frame)
    cerebro.adddata(data)

    cerebro.optstrategy(
        strategy,
        target_percent=target_percent,
        margin_per_contract=margin,
        printlog=printlog,
    )

    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    back = cerebro.run()

    par_list = [
        [
            x[0].params.target_percent,
            x[0].analyzers.returns.get_analysis()["rnorm100"],
            x[0].analyzers.sharpe.get_analysis()["sharperatio"],
            x[0].analyzers.drawdown.get_analysis()["max"]["drawdown"],
        ]
        for x in back
    ]

    par_df = pd.DataFrame(
        par_list, columns=["target_percent", "return", "sharpe", "dd"]
    )

    return par_df.round(2)


def run_backtest(
    frame: pd.DataFrame,
    strategy: bt.Strategy = SignalStrategy,
    initial_cash: float = 100_000.0,
    commission: float = 2.5,
    margin: int = 10000,
    mult: int = 25000,
    commtype: int = bt.CommInfoBase.COMM_FIXED,
    slippage: float = 0.0015,  # 3 ticks of 0.0005 each
    target_percent: float = 0.98,
    printlog: bool = False,
):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(
        commission=commission, margin=margin, mult=mult, commtype=commtype
    )
    cerebro.broker.set_slippage_fixed(fixed=slippage)

    data = SignalData(dataname=frame)
    cerebro.adddata(data)

    cerebro.addstrategy(
        strategy,
        target_percent=target_percent,
        margin_per_contract=margin,
        printlog=printlog,
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days
    )

    print(f"Starting equity: {cerebro.broker.getvalue():,.2f}")
    results = cerebro.run()
    strat = results[0]
    print(f"Ending equity:   {cerebro.broker.getvalue():,.2f}")

    time_return = strat.analyzers.time_return.get_analysis()
    returns = pd.Series(time_return).sort_index()
    returns.index = pd.to_datetime(returns.index)
    returns.name = "strategy"

    fig = cerebro.plot(style="bar")[0][0]
    fig.set_size_inches(10, 6)
    fig.set_dpi(300)

    return {
        "returns": returns,
        "fig": fig,
    }
