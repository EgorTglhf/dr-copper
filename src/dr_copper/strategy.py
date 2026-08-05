import math

import backtrader as bt


class SignalData(bt.feeds.PandasData):
    lines = ("signal",)
    params = (("signal", -1),)


class SignalStrategy(bt.Strategy):
    params = dict(
        target_percent=0.95,
        printlog=False,
        margin_per_contract=10000,
    )

    def log(self, txt, dt=None):
        if self.p.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print("%s, %s" % (dt.isoformat(), txt))

    def __init__(self):
        self.signal = self.datas[0].signal
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Completed]:
            side = "BUY" if order.isbuy() else "SELL"
            self.log(
                f"{side} EXECUTED, price {order.executed.price:.4f}, "
                f"size {order.executed.size:.2f}, "
                f"comm {order.executed.comm:.2f}"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("Order Canceled/Margin/Rejected")

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log("OPERATION PROFIT, GROSS %.2f, NET %.2f" % (trade.pnl, trade.pnlcomm))

    def next(self):
        if self.order:
            return

        sig = self.signal[0]
        if math.isnan(sig):
            return

        pos_size = self.getposition().size
        equity = self.broker.getvalue()
        contracts = int((equity * self.p.target_percent) // self.p.margin_per_contract)

        if sig > 0 and pos_size <= 0:
            self.order = self.order_target_size(target=contracts)  # go long N contracts
        elif sig < 0 and pos_size >= 0:
            self.order = self.order_target_size(
                target=-contracts
            )  # go short N contracts
        elif sig == 0 and pos_size != 0:
            self.order = self.order_target_size(target=0)


class SMAStrategy(SignalStrategy):
    params = dict(
        target_percent=0.95,
        sma_slow=200,
        sma_fast=50,
        printlog=False,
        margin_per_contract=10000,
    )

    def __init__(self):
        super().__init__()
        sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.sma_slow
        )
        sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.sma_fast
        )

        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)

    def next(self):
        if self.order:
            return

        pos_size = self.getposition().size
        equity = self.broker.getvalue()
        contracts = int((equity * self.p.target_percent) // self.p.margin_per_contract)

        if self.crossover > 0 and pos_size <= 0:
            self.order = self.order_target_size(target=contracts)  # go long N contracts
        elif self.crossover < 0 and pos_size >= 0:
            self.order = self.order_target_size(
                target=-contracts
            )  # go short N contracts


class SignalSMAStrategy(SignalStrategy):
    params = dict(
        target_percent=0.95,
        sma_slow=200,
        sma_fast=50,
        printlog=False,
        margin_per_contract=10000,
    )

    def __init__(self):
        super().__init__()
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.sma_slow
        )
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.p.sma_fast
        )

    def next(self):
        self.log(
            f"TEST, {self.data.datetime.date(0)},"
            f"signal {self.signal[0]}, "
            f"size {self.getposition().size}, "
            f"order {self.order}"
        )

        if self.order:
            return

        sig = self.signal[0]
        if math.isnan(sig):
            return

        pos_size = self.getposition().size
        equity = self.broker.getvalue()
        contracts = int((equity * self.p.target_percent) // self.p.margin_per_contract)

        if sig > 0 and pos_size <= 0 and self.sma_fast > self.sma_slow:
            self.order = self.order_target_size(target=contracts)  # go long N contracts
        elif sig < 0 and pos_size >= 0 and self.sma_fast < self.sma_slow:
            self.order = self.order_target_size(
                target=-contracts
            )  # go short N contracts
        elif sig == 0 and pos_size != 0:
            self.order = self.order_target_size(target=0)

    ## doesn't work for futures because of order_target_percent abs(size)
    # def next(self):
    #     self.log(f'TEST, {self.data.datetime.date(0)},'
    #                   f'signal {self.signal[0]}, '
    #                   f'size {self.getposition().size}, '
    #                   f'order {self.order}')
    #     if self.order:
    #         return  # one order in flight at a time
    #     sig = self.signal[0]
    #     if math.isnan(sig):
    #         return
    #     pos_size = self.getposition().size
    #     if sig > 0:
    #         if pos_size <= 0:
    #             self.order = self.order_target_percent(target=self.p.target_percent)
    #     elif sig < 0:
    #         if pos_size >= 0:
    #             self.order = self.order_target_percent(target=-self.p.target_percent)
    #     else:
    #         if pos_size != 0:
    #             self.order = self.order_target_percent(target=0.0)
