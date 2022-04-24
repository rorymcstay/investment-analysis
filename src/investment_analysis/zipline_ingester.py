""""""
import threading
import logging
import hashlib
from datetime import datetime
from typing import Dict

import pytz
import toolz
from progressbar import progressbar

import yfinance as yf
import pandas as pd
import numpy as np

from zipline.data.minute_bars import BcolzMinuteBarWriter
from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter
from zipline.data.adjustments import SQLiteAdjustmentWriter
from zipline.data import bundles
from zipline.utils.calendar_utils import TradingCalendar
from zipline.assets import AssetDBWriter, Asset

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.engine import SimplePipelineEngine



logger = logging.getLogger(__name__)




PERIOD_YEARS = 5

def df_cache_key(name, period, symbol):
    now = datetime.now()
    return f'{name}_{period}_{symbol}_{now.year}_{now.month}_{now.day}'


def _yield_yahoo_finance_data(
        equities: pd.DataFrame,
        universe: yf.Tickers,
        cache: Dict[str, pd.DataFrame],
        calendar: TradingCalendar,
    ):

    for asset in equities.itertuples():
        ticker = universe.tickers[asset.symbol]
        symbol = asset.symbol
        cache_key = df_cache_key('history', PERIOD_YEARS, symbol)
        data = ticker.history(f'{PERIOD_YEARS}y') if cache_key not in cache else cache[cache_key]
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        ohlc: pd.DataFrame = data[['open', 'high', 'low', 'close', 'volume']]
        sessions = calendar.sessions_in_range(asset.start_date, asset.end_date)
        ohlc_clean = ohlc.dropna()
        ohlc_clean.index = ohlc_clean.index.tz_localize(pytz.utc)
        ohlc_clean = ohlc_clean.reindex(sessions)
        indexer = sessions.slice_indexer(asset.start_date, asset.end_date)

        yield (asset.Index,ohlc_clean)


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def _split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def _batch_info(tickers, batch_size):

    def get_info(ticker, storage):
        try:
            storage.append((ticker, ticker.get_info()))
        except Exception:
            logger.error("Error retrieving info for %s", ticker.ticker)

    for batch in progressbar(list(_chunks(tickers, min(len(tickers), batch_size))), prefix=f'{batch_size=}'):
        info_batch = []
        threads = [threading.Thread(target=get_info, args=(t, info_batch)) \
                                                                for t in batch]
        [t.start() for t in threads]
        [t.join() for t in threads]
        yield info_batch


def yahoo_finance(tickers, batch_size=10):
    return lambda *args, **kwargs: _yahoo_finance(*args, **kwargs,
                                            tickers=tickers,
                                            batch_size=batch_size)


def _yahoo_finance(environ: Dict[str, str],
       asset_db_writer: AssetDBWriter,
       minute_bar_writer: BcolzMinuteBarWriter,
       daily_bar_writer: BcolzDailyBarWriter,
       adjustment_writer: SQLiteAdjustmentWriter,
       calendar: TradingCalendar,
       start_session: pd.Timestamp,
       end_session: pd.Timestamp,
       cache: Dict[str, pd.DataFrame],
       show_progress: bool,
       output_dir,
       tickers=['AAPL'],
       batch_size=10,):
    """"""

    universe = yf.Tickers(list(tickers))
    assets = []
    info_cache_key = hashlib.sha224(f'{PERIOD_YEARS}y_{str(" ".join(tuple(k for k in universe.tickers.keys())))}'.encode('utf-8')).hexdigest()
    if info_cache_key not in cache:
        ticker: yf.Ticker
        for batch in _batch_info(list(universe.tickers.values()), batch_size=batch_size):
            for ticker, info in batch:
                if info.get('regularMarketPrice') is None:
                    continue
                key = df_cache_key('history', PERIOD_YEARS, ticker.ticker)
                history = cache[key] if key in cache else ticker.history(f'{PERIOD_YEARS}y')
                cache[key] = history
                if history.empty:
                    continue
                assets.append({
                    'sid': tickers.index(ticker.ticker),
                    'symbol': ticker.ticker,
                    'asset_name': info['shortName'],
                    'exchange': info['exchange'],
                    'start_date': history.iloc[[0]].index[0],
                    'first_traded': history.iloc[[0]].index[0],
                    'end_date': history.iloc[[-1]].index[0]
                })
        equities = pd.DataFrame(data=assets)
    else:
        equities = cache[info_cache_key]
    cache[info_cache_key] = equities
    equities.index = equities.sid
    print(equities)
    asset_db_writer.write(equities=equities, chunk_size=100)

    daily_bar_writer.write(_yield_yahoo_finance_data(
            equities, universe, cache, calendar
        ), invalid_data_behavior='warn')

    for symbol, ticker in universe.tickers.items():
        key = df_cache_key('history', PERIOD_YEARS, symbol)
        data = cache[key] if key in cache else ticker.history(f'{PERIOD_YEARS}y')
        if data.empty:
            continue
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        stock_splits: pd.DataFrame = data[data.stock_splits != 0.0][['stock_splits']].dropna()
        dividends: pd.Series = data[data['dividends'] > 0]['dividends'].squeeze().dropna()
        if not stock_splits.empty:
            stock_splits['sid'] = VANGUARD_UNIVERSE[symbol]
            stock_splits['ratio'] = stock_splits.stock_splits
            stock_splits['effective_date'] = stock_splits.index.view(np.int64)
            stock_splits = stock_splits[['ratio', 'sid', 'effective_date']]
            stock_splits = stock_splits.reset_index(drop=True)
            stock_splits.index = stock_splits.sid
        if not dividends.empty:
            dividends_ = pd.DataFrame()
            dividends_['pay_date'] = dividends.index
            dividends_['ex_date'] = dividends.index
            dividends_['declared_date'] = dividends.index
            dividends_['record_date'] = dividends.index
            dividends_['amount'] = dividends.values
            dividends_['sid'] = VANGUARD_UNIVERSE[symbol]
            dividends=dividends_
        adjustment_writer.write(splits=stock_splits if not stock_splits.empty else None,
                                dividends=dividends if not dividends.empty else None)



@toolz.memoize
def _pipeline_engine_and_calendar_for_bundle(bundle):
    """Create a pipeline engine for the given bundle.

    Parameters
    ----------
    bundle : str
        The name of the bundle to create a pipeline engine for.

    Returns
    -------
    engine : zipline.pipleine.engine.SimplePipelineEngine
        The pipeline engine which can run pipelines against the bundle.
    calendar : zipline.utils.calendars.TradingCalendar
        The trading calendar for the bundle.
    """
    bundle_data = bundles.load(bundle)
    pipeline_loader = USEquityPricingLoader.without_fx(
        bundle_data.equity_daily_bar_reader,
        bundle_data.adjustment_reader,
    )

    def choose_loader(column):
        if column in USEquityPricing.columns:
            return pipeline_loader
        raise ValueError(
            'No PipelineLoader registered for column %s.' % column
        )

    calendar = bundle_data.equity_daily_bar_reader.trading_calendar
    return (
        SimplePipelineEngine(
            choose_loader,
            bundle_data.asset_finder,
            #calendar.all_sessions,
        ),
        calendar,
    )


def run_pipeline_against_bundle(pipeline, start_date, end_date, bundle):
    """Run a pipeline against the data in a bundle.

    Parameters
    ----------
    pipeline : zipline.pipeline.Pipeline
        The pipeline to run.
    start_date : pd.Timestamp
        The start date of the pipeline.
    end_date : pd.Timestamp
        The end date of the pipeline.
    bundle : str
        The name of the bundle to run the pipeline against.

    Returns
    -------
    result : pd.DataFrame
        The result of the pipeline.
    """
    engine, calendar = _pipeline_engine_and_calendar_for_bundle(bundle)

    start_date = pd.Timestamp(start_date, tz='utc')
    if not calendar.is_session(start_date):
        # this is not a trading session, advance to the next session
        start_date = calendar.minute_to_session_label(
            start_date,
            direction='next',
        )

    end_date = pd.Timestamp(end_date, tz='utc')
    if not calendar.is_session(end_date):
        # this is not a trading session, advance to the previous session
        end_date = calendar.minute_to_session_label(
            end_date,
            direction='previous',
        )

    return engine.run_pipeline(pipeline, start_date, end_date)

if __name__ == '__main__':
    from zipline.pipeline import Pipeline
    from zipline.pipeline.data import USEquityPricing
    out = run_pipeline_against_bundle(
       Pipeline({'close': USEquityPricing.close.latest}),
       '2012',
       '2013',
       bundle='quandl'
    )
    print(out)
