import logging
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pandas_datareader.data as web

from zipline import api
from zipline import protocol
from zipline import run_algorithm

from investment_analysis.markowitz import computation
from investment_analysis.zipline_ingester import VANGUARD_UNIVERSE


logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def initialize(context):
    context.tick = 0
    context.window_size = 300
    context.rebal_interval = 30
    context.price_col = 'close'
    context.assets = [api.symbol(ticker) for ticker in VANGUARD_UNIVERSE.keys()]

def handle_data(context, data: protocol.BarData):
    # Allow history to accumulate 100 days of prices before trading
    # and rebalance every day thereafter.
    context.tick += 1
    if context.tick < context.window_size:
        return
    if context.tick % context.rebal_interval != 0:
        # rebalance every 30 days
        return
    # Get rolling window of past prices and compute returns
    prices = data.history(
            assets=list(filter(data.can_trade, context.assets)),
            bar_count=context.window_size,
            fields=[context.price_col],
            frequency='1d')[context.price_col]
    returns = prices \
            .pct_change() \
            .dropna() \
            .unstack() \
            .dropna()
    # Perform Markowitz-style portfolio optimization
    try:
        weights, _, _ = computation.optimal_portfolio(returns.T)
    except ValueError as ex:
        logger.warning("failed to compute optimal portfolio %s", ex.args)
        return
    # Rebalance portfolio accordingly
    out = {}
    weights = np.array([w[0] for w in weights])
    normalized_weights = weights/sum(weights)
    for stock, weight in zip(returns.columns, normalized_weights):
        out[stock.symbol] = weight
        api.order_target_percent(stock, weight)
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
            bundle='vanguard-etf-universe',
            data_frequency='daily'
    )

    result.to_pickle('./backtest.pickle')
