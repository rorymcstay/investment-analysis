""""""
from datetime import datetime
import logging
from typing import Dict

import yfinance as yf
import pandas as pd
import numpy as np

from zipline.data.minute_bars import BcolzMinuteBarWriter
from zipline.data.bcolz_daily_bars import BcolzDailyBarWriter
from zipline.data.adjustments import SQLiteAdjustmentWriter
from zipline.utils.calendar_utils import TradingCalendar
from zipline.assets import AssetDBWriter, Asset


logger = logging.getLogger(__name__)


VANGUARD_UNIVERSE = {
    'VUKE.L': 1,
    'VMID.L': 2,
    'VUSA.L': 3,
    'VERX.L': 4,
    'VGER.L': 6,
    'VWRL.L': 8,
    'VHYL.L': 9,
    'VEVE.L': 10,
    'VJPN.L': 11,
    'VAPX.L': 12,
    'VNRT.L': 13,
    'VFEM.L': 14,
    'VECP.L': 15,
    'VETY.L': 16,
    'VAGP.L': 17,
    'VGOV.L': 18,
    'VUCP.L': 19,
    'VUTY.L': 20,
    'VEMT.L': 21,
}

PERIOD = '5y'

def df_cache_key(name, period, symbol):
    now = datetime.now()
    return f'{name}_{period}_{symbol}_{now.year}_{now.month}_{now.day}'


def _yield_asset_from_yfinance(universe,
        adjustment_writer: SQLiteAdjustmentWriter,
        cache: Dict[str, pd.DataFrame]):
    ticker: yf.Ticker

    for symbol, ticker in universe.tickers.items():
        cache_key = df_cache_key('history', PERIOD, symbol)
        data = ticker.history(PERIOD) if cache_key not in cache else cache[cache_key]
        sid = VANGUARD_UNIVERSE[symbol]
        data.columns = [c.lower().replace(' ', '_') for c in data.columns]
        ohlc = data[['open', 'high', 'low', 'close', 'volume']]
        stock_splits: pd.DataFrame = data[data.stock_splits != 0.0][['stock_splits']]
        stock_splits['sid'] = sid
        stock_splits['ratio'] = stock_splits.stock_splits
        stock_splits['effective_date'] = stock_splits.index.astype(np.int64)
        stock_splits = stock_splits[['ratio', 'sid', 'effective_date']]
        stock_splits = stock_splits.reset_index(drop=True)
        dividends = data[data['dividends'] > 0]['dividends']
        dividends_ = pd.DataFrame()
        dividends.index = dividends.index
        dividends_['pay_date'] = dividends.index
        dividends_['ex_date'] = dividends.index
        dividends_['declared_date'] = dividends.index
        dividends_['record_date'] = dividends.index
        dividends_['amount'] = dividends
        adjustment_writer.write(splits=stock_splits, dividends=dividends_)
        yield (sid, ohlc)


def vanguard_universe():

    def _ingest_yfinance(environ: Dict[str, str],
           asset_db_writer: AssetDBWriter,
           minute_bar_writer: BcolzMinuteBarWriter,
           daily_bar_writer: BcolzDailyBarWriter,
           adjustment_writer: SQLiteAdjustmentWriter,
           calendar: TradingCalendar,
           start_session: pd.Timestamp,
           end_session: pd.Timestamp,
           cache: Dict[str, pd.DataFrame],
           show_progress: bool,
           output_dir):
        """"""
        universe = yf.Tickers(list(VANGUARD_UNIVERSE.keys()))
        assets = []
        info_cache_key = str(" ".join(tuple(k for k in universe.tickers.keys())))
        if info_cache_key not in cache:
            ticker: yf.Ticker
            for symbol, ticker in universe.tickers.items():
                info = ticker.get_info()
                key = df_cache_key('history', PERIOD, symbol)
                history = cache.get(key) or ticker.history(PERIOD)
                cache[key] = history
                assets.append({
                    'sid': VANGUARD_UNIVERSE[symbol],
                    'symbol': symbol,
                    'asset_name': info['longName'],
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
        asset_db_writer.write(equities=equities)
        daily_bar_writer.write(
                _yield_asset_from_yfinance(
                    universe=universe,
                    adjustment_writer=adjustment_writer,
                    cache=cache),
                show_progress=True,
                invalid_data_behavior='raise'
                )
    return _ingest_yfinance
