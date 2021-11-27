# load data
import sqlite3
import pandas as pd
from datetime import datetime
import pickle
import json

con = sqlite3.connect('./example.db')

ftse_prediction = pd.read_sql('select * from ftse_prediction', con)
test_cols = list(filter(lambda col: col not in ['level_0', 'index', 'target'],
                        ftse_prediction.columns))


ftse_prediction.index = pd.to_datetime(ftse_prediction['level_0'])
ftse_prediction = ftse_prediction[['target', 'index'] + test_cols]

train_start = pd.to_datetime('30/03/2016')
train_end = pd.to_datetime('31/01/2021')

test_mask = lambda s: (s.index > train_end) & (s.index <= datetime.now())
train_mask = lambda s: (s.index > train_start) & (s.index <= train_end)

test = ftse_prediction[test_mask(ftse_prediction)]
X_test = test[test_cols]
y_test = test['index']

train = ftse_prediction[train_mask(ftse_prediction)]
X_train = train[test_cols]
y_train = train['index']
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': 10000,
    'learning_rate': 0.01,
    'max_features': 30,
    'max_depth': 4,
    'random_state': 0
}

parameter_space = {
    'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'max_depth': [6,8,10],
    'min_samples_split': [2,5,10],
    'n_estimators': [2000,3000,4000,5000]
}

#base_estimator = GradientBoostingRegressor()

base_learner = GradientBoostingRegressor(max_features=60)
sh = GridSearchCV(base_learner, parameter_space, n_jobs=8, pre_dispatch=16)

if __name__ == '__main__':
    sh.fit(X_train, y_train)
    with open('./lib/models/ftse_prediction_cv.pickle', 'wb') as picklefile:
        picklefile.write(pickle.dumps(sh))
    with open('./etc/models/ftse_predcition_cv_grid.json', 'w') as jsonfile:
        jsonfile.write(json.dumps(parameter_space))

