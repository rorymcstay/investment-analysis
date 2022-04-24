import multiprocessing
import time
import json
from sklearn.ensemble import GradientBoostingRegressor
import time
from tqdm.notebook import tqdm as tqdm_notebook
import pandas as pd
import sqlite3
import pickle
import os
from datetime import datetime


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

y_train = train['index']

lr_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
max_features = [60]
max_depths=[40]#, 50, 60, 70, 80, 90, 100]
n_estimatorss=[1000, 2000,3000,4000,5000]
manager = multiprocessing.Manager()
results = manager.dict()

results['models'] = manager.list()
results['threads'] = manager.dict()
results['trained'] = 0
results['minimum'] = -1000
results['maximum'] = 0

parameter_space = {
    'learning_rate': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
    'max_depth': [6,8,10],
    'min_samples_split': [2,5,10],
    'n_estimators': [2000,3000,4000,5000]
}

def train_model_and_print_score(modelClass, X, y, X_test, y_test, **kwargs):
    model = modelClass(**kwargs)
    model.fit(X, y)
    results['trained'] += 1
    score = model.score(X_test, y_test)
    pid = os.getpid()
    results['threads'][pid] = f'pid={pid}, params={kwargs}, score={score}'
    if score  > results['minimum']:
        if len(results['models']) <= 5:
            results['models'].append(model)
        else:
            results['models'][-1] = model
            models = sorted(models, key=lambda model: model.score(X_test, y_test), reverse=True)
            results['models'] = models
            results['minimum'] = results['models'][-1].score(X_test, y_test)
            results['maximum'] = results['models'][0].score(X_test, y_test)
        #print(f'model-{results["trained"]}: test_score={model.score(X_test, y_test)}, params={kwargs}', flush=True, end='\r')

params = []

for learning_rate in lr_list:
    for max_feature in max_features:
        for max_depth in max_depths:
            for n_estimators in n_estimatorss:
                params.append(dict(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_features=max_feature,
                                    max_depth=max_depth,
                                    random_state=None))
if __name__ == '__main__':
    print(X_train.head())
    print(y_train.head())
    models=[]
    to_print=[]
    time_begin=time.time()
    with multiprocessing.Pool(processes=8) as pool:
        args = (GradientBoostingRegressor, X_train, y_train, X_test, y_test)
        try:
            asyncresults = [pool.apply_async(train_model_and_print_score, args=args, kwds=param) for param in params]
            pool.close()
            while not all(result.ready() for result in asyncresults):
                time.sleep(1)
                print(f'max={results["maximum"]} min={results["minimum"]} trained={results["trained"]}/{len(params)}', end='\r')
                #[results['threads'][pid] for pid in results.get('threads', {})]
        except KeyboardInterrupt as ex:
            pass

    count = 0
    elapsed = (time.time() - time_begin)/60
    print(f'\ncount={len(results["models"])}, elapsed={elapsed}mins')

    for model in sorted(results['models'], key=lambda model: model.score(X_test, y_test), reverse=True):
        count += 1
        mdl = model
        score = mdl.score(X_test, y_test)
        if count > 5:
            break
        with open(f'lib/models/ftse_prediction_model_{count}.pickle', 'wb') as model_file:
            model_file.write(pickle.dumps(mdl))
        with open(f'etc/models/ftse_prediction_model_{count}.summary.json', 'w') as model_summary:
            model_summary.write(json.dumps({"params": mdl.get_params(), "score": score, "inpute": list(X_test.columns)}))
        print(f'score={mdl.score(X_test, y_test)}, params={mdl.get_params()}')

