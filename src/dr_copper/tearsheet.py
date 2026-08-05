import pandas as pd
import quantstats as qs

qs.extend_pandas()


def benchmark_returns_from_prices(prices: pd.DataFrame) -> pd.Series:
    bench = prices["close"].pct_change().dropna()
    bench.name = "benchmark"
    return bench


def print_key_metrics(returns: pd.Series, benchmark: pd.Series = None):
    print("=" * 60)
    print("KEY METRICS — strategy")
    print("=" * 60)
    print(f"Sharpe            : {qs.stats.sharpe(returns):.3f}")
    print(f"Sortino           : {qs.stats.sortino(returns):.3f}")
    print(f"Calmar            : {qs.stats.calmar(returns):.3f}")
    print(f"CAGR              : {qs.stats.cagr(returns):.2%}")
    print(f"Max Drawdown      : {qs.stats.max_drawdown(returns):.2%}")
    print(f"Volatility (ann.) : {qs.stats.volatility(returns):.2%}")
    print(f"Win rate          : {qs.stats.win_rate(returns):.2%}")
    print(f"Profit factor     : {qs.stats.profit_factor(returns):.3f}")

    if benchmark is not None:
        print("-" * 60)
        print("KEY METRICS — benchmark (buy & hold HG=F)")
        print("-" * 60)
        print(f"Sharpe            : {qs.stats.sharpe(benchmark):.3f}")
        print(f"Calmar            : {qs.stats.calmar(benchmark):.3f}")
        print(f"CAGR              : {qs.stats.cagr(benchmark):.2%}")
        print(f"Max Drawdown      : {qs.stats.max_drawdown(benchmark):.2%}")


def generate_html_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series = None,
    output_path: str = "Phase_5_tearsheet.html",
    title: str = "Dr.Copper Phase 5 — Strategy Tearsheet",
):
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=output_path,
        title=title,
    )
    print(f"Tearsheet written to {output_path}")
