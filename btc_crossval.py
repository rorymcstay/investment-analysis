from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from tick_funcs import process, tickdata_on, transformations, get_test_train

df = process(
    tickdata_on(day=1, month=6, year=2021),
    tickdata_on(day=2, month=6, year=2021),
    tickdata_on(day=3, month=6, year=2021),
    tickdata_on(day=4, month=6, year=2021),
    tickdata_on(day=5, month=6, year=2021),
    # enrich with features
    **transformations
)

features = ['bidSize', 'bidPrice', 'askSize', 'askPrice', 'timesince', 'prev_bidSize', 'prev_askSize']
target = ['midPointDelta']


print(df[features])

X = df[features].values
y = df[target].values

X_test, X_train, y_test, y_train = get_test_train(X, y)

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
#gb_reg.fit(X_train, y_train)
base_learner = GradientBoostingRegressor()
sh = GridSearchCV(base_learner, parameter_space, n_jobs=7, pre_dispatch=7)

if __name__ == '__main__':
    sh.fit(X_train, y_train)
    with open('./lib/models/XBTUSD_tick.pickle', 'wb') as picklefile:
        picklefile.write(pickle.dumps(sh))
    with open('./etc/models/XBTUSD_tick.json', 'w') as jsonfile:
        jsonfile.write(json.dumps(parameter_space))


