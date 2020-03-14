import os
import numpy as np
import regressor as reg
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt


if __name__ == "__main__":
    mol_type = 'atp'
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_regression_feature_vector_without_tag_mar2020.csv' % mol_type)

    assert os.path.isfile(data_file)
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape

    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1]
    yu = np.unique(y)
    ye = y.copy()

    max_dev = 0.2
    test_split = 0.4
    best_data, score, max_err, best_model = None, None, None, None
    epochs, early_stop = 1000, 50
    yu_actual = np.arctanh(yu)*200 + 100
    counter, epoch = 0, 0
    last_error = np.zeros(n)

    while (epoch < epochs) and (counter < early_stop):
        predictor_model = reg.XGBoost(booster='gbtree',
                                      max_depth=4,
                                      early_stopping_rounds=10)

        corrector_model = reg.XGBoost(booster='gbtree',
                                      max_depth=4,
                                      early_stopping_rounds=10)

        acc = predictor_model.fit(x, ye, test_split=test_split)
        err = np.clip(y - predictor_model.predict(x),
                      a_min=-max_dev, a_max=max_dev)

        ye = np.clip(y - err, a_min=-1.0, a_max=1.0)
        corrector_model.fit(x, err, test_split=test_split)

        yp = predictor_model.predict(x) + corrector_model.predict(x)
        curr_error = y - yp
        max_error = np.max(np.abs(curr_error - last_error))
        mad = mean_absolute_error(y, yp)
        last_error = curr_error

        if (score is None) or (score > mad) or (max_err is None) or (max_error < max_err):
            score = mad
            max_err = max_error
            best_model = {'predictor': predictor_model,
                          'corrector': corrector_model}
            counter = 0
        counter = counter + 1
        epoch = epoch + 1
        print('%d: %.5f %.5f' % (epoch, max_error, mad))



