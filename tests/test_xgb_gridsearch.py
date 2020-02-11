import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scripts', 'output')
    assert os.path.isdir(data_dir)
    amino = 'leu'
    data_file = os.path.join(data_dir, '%s_sasa_features.csv' % amino.lower())
    assert os.path.isfile(data_file)

    with open(data_file, "r") as fp:
        data = np.loadtxt(fp, delimiter=',')
    n, w = data.shape
    x = data[:, :-1].reshape((n, w-1))
    y = data[:, -1].reshape((n, 1))

    learning_rates = np.linspace(0.1, 0.3, 3).tolist()
    booster = ['gbtree', 'dart', 'gblinear']
    depth = list(range(2, 7, 2))
    child_wt = list(range(1, 7, 2))
    sub_sample=np.linspace(0.5, 0.9, 3)
    sample_by_tree = np.linspace(0.5, 0.7, 3).tolist()
    n_estimators = list(range(150, 300, 50))

    model = xgb.XGBRegressor(objective='reg:squarederror',
                             booster='gbtree',
                             learning_rates=0.2,
                             min_child_weight=5,
                             n_estimators=250,
                             colsample_bytree=0.7,
                             max_depth=6,
                             nthread=2)
    param_grid = {
                 # "learning_rate": learning_rates,
                 # "max_depth": depth,
                 # "booster": booster,
                  "subsample": sub_sample,
                 # "colsample_bytree": sample_by_tree,
                 # "n_estimators": n_estimators
                 }
    kfold = KFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model,
                               param_grid,
                               scoring="r2",
                               cv=kfold,
                               n_jobs=2,
                               verbose=5)
    result = grid_search.fit(x, y)
    print(result.best_score_)
    print(result.best_params_)

