import logging
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader.data as web

from zipline import api
from zipline import protocol
from zipline import run_algorithm
from zipline.algorithm import TradingAlgorithm

from investment_analysis.markowitz import computation
from investment_analysis.zipline_ingester import VANGUARD_UNIVERSE


logger = logging.getLogger(__name__)

def initialize(context: TradingAlgorithm):
    context.tick = 0
    context.window_size = 100
    context.rebal_interval = 30
    context.price_col = 'close'
    context.traded_assets = list(map(api.symbol, [
        'VUKE.L',
        'VMID.L',
        'VUSA.L',
        'VGER.L',
    ]))
    context.benchmarks = list(map(api.symbol, [
        '^FTSE',
        '^FTMC',
        '^GSPC',
        '^GDAXI',
    ]))
    assert len(context.benchmarks) == len(context.traded_assets)

def handle_data(context: TradingAlgorithm, data: protocol.BarData):
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every rebal_interval days thereafter.
    context.tick += 1
    if context.tick < context.window_size:
        return
    if context.tick % context.rebal_interval != 0:
        # rebalance every 30 days
        return
    # Get rolling window of past prices and compute returns
    benchmark_prices = data.history(
            assets=list(filter(data.can_trade, context.benchmarks)),
            bar_count=context.window_size,
            fields=[context.price_col],
            frequency='1d')[context.price_col]
    prices = data.history(
            assets=list(filter(data.can_trade, context.traded_assets)),
            bar_count=context.window_size,
            fields=[context.price_col],
            frequency='1d')[context.price_col]


    returns: pd.DataFrame = prices \
                .pct_change() \
                .unstack() \
                .dropna()
    benchmark_returns: pd.DataFrame = benchmark_prices \
                .pct_change() \
                .unstack() \
                .dropna()
    benchmark_value = benchmark_returns.add(1).cumsum().iloc[[-1]]
    traded_value = returns.add(1).cumsum().iloc[[-1]]
    for bench, asset in zip(context.traded_assets, context.benchmarks):
        tracking_error = (benchmark_value[bench] - traded_value[asset]).iloc[[-1]]
        out['tracking_error_{asset.symbol}'] = tracking_error
    #for stock, benchmark, weight in zip(returns.columns, benchmark_returns.columns, normalized_weights):
    #    diff = benchmark_returns[benchmark].period_value - returns[stock].period_value
    #    out[stock.symbol] = weight
    #    api.order_target_percent(stock, weight)
    api.record(**out)


if __name__ == '__main__':

    start = pd.Timestamp('2017-04-10')
    end = pd.Timestamp('2022-04-08')
    logger.setLevel(logging.INFO)

    sp500 = web.DataReader('SP500', 'fred', start, end).SP500
    benchmark_returns = sp500.pct_change()

    result: pd.DataFrame = run_algorithm(
            start=start.tz_localize('UTC'),
            end=end.tz_localize('UTC'),
            initialize=initialize,
            handle_data=handle_data,
            #analyze=analyze,
            benchmark_returns=benchmark_returns,
            capital_base=27000,
            bundle='yahoo-finance-universe',
            data_frequency='daily'
    )

    result.to_pickle('./tracking_error/backtest.pickle')
