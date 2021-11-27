from datetime import datetime
import pandas as pd

def days_in_month(month: int) -> int:
    data = {
        1:31,
        2:28,
        3:31,
        4:30,
        5:31,
        6:30,
        7:31,
        8:31,
        9:30,
        10:31,
        11:30,
        12:31
    }
    return data(month)


def tickdata_on(day=datetime.now().day, month=datetime.now().month, year=datetime.now().year, symbol = 'XBTUSD') -> pd.DataFrame:
    datestr = f'{day}-{month}-{year}'
    base_dir = '/home/rory/dev/tick-capture'
    df = pd.read_csv(f'{base_dir}/{datestr}/{symbol}_tick.csv')
    return df

def process(*argsDf : pd.DataFrame, **kwargs):
    out = pd.DataFrame()
    for tickData in argsDf:
        out = out.append(tickData)

    for feature in kwargs:
        print(f'Applying feature {feature}')
        func = kwargs.get(feature)
        out[feature] = func(out)
    out.index = pd.DatetimeIndex(out.timestamp, tz='utc')

    return out

def time_since(df):
    df['timesince'] = pd.DatetimeIndex(df.index) - pd.DatetimeIndex(df.timestamp.shift(1))
    df['timesince'] = df.timesince.apply(lambda val: val.microseconds)
    return df['timesince']

def get_test_train(X, y):
    train_size = int(len(X) * 0.60)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(X)]
    print('Observations: %d' % (len(X)))
    print('Training Observations: %d' % (len(X_train)))
    print('Testing Observations: %d' % (len(X_test)))
    return X_test, X_train, y_test, y_train

transformations = dict(
    #prev_bidSize=lambda df: df['bidSize'].shift(1),
    #prev_askSize=lambda df: df['askSize'].shift(1),
    midPoint=lambda df: (df.askPrice + df.bidPrice)/2,
    midPointDelta=lambda df: df.midPoint - df.midPoint.shift(1),
    timesince=lambda df:pd.Series(pd.DatetimeIndex(df.timestamp)-pd.DatetimeIndex(df.timestamp.shift(1)), index=df.index).apply(lambda val: val.microseconds),
)

