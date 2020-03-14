import os
import numpy as np
import pandas as pd
import regressor as reg
from matplotlib import pyplot as plt


if __name__ == "__main__":
    mol_type = 'atp'
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_regression_feature_vector_with_tag_mar2020.csv' % mol_type)
    assert os.path.isfile(data_file)
    data = pd.read_csv(data_file, delimiter=',', header=None)

    data_tag = data.iloc[:, 0].tolist()

    def check_snapshot(x):
        if "cryst" in x:
            return True
        tag = x.split("_")[2][0]
        start, end = float(x.split("_")[2][1:]) , float(x.split("_")[3])
        return (end - start) <= 1

    valid_rows = pd.Series([check_snapshot(x) for x in data_tag])
    print("Before filtering: (%d, %d)" % data.shape)
    data = data[valid_rows].iloc[:, 1:].values
    print("After filtering: (%d, %d)" % data.shape)

    np.random.shuffle(data)
    n, f = data.shape
    eps = 1e-6

    def rev_transform(v):
        return np.arctanh(v) * 200 + 100

    def transform(v):
        return np.tanh((v - 100)/200)

    max_dev_list = [10, 20, 30, 40, 50]
    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1]
    y_actual = rev_transform(y)
    yu = np.unique(y)

    for max_dev in max_dev_list:
        bootstrap, epochs, test_split, early_stop = 10, 5000, 0.3, 10
        run_avg, acc_avg = 20, None
        best_data, score, best_model = None, None, None
        counter, iteration, ds = 0, 0, 0.5
        err = np.zeros(y.shape[0])
        sgn = np.zeros(y.shape[0])
        accepted = False

        fig = plt.figure(1)
        ax = fig.add_subplot(1, 1, 1)
        accuracy_run, h = [], None
        plt.ion()
        plt.show()
        y_lo = np.clip(y_actual - max_dev, a_min=0, a_max=np.inf)
        y_hi = y_actual + max_dev
        while (iteration < epochs) and (counter < early_stop):
            if iteration > 0:
                sgn = np.sign(err)
                err = ds * sgn
                ye_last = rev_transform(ye)
            else:
                ye_last = y_actual
            ye_new = np.clip(ye_last + err, a_min=y_lo, a_max=y_hi)
            ye_dev = ye_new - y_actual
            print("@ %d iteration: shift {mean:%.2f, sd: %.2f}" %
                  (iteration, np.mean(ye_dev), np.std(ye_dev)))
            ye = np.clip(transform(ye_new), a_min=-1, a_max=1)

            if h is not None:
                h[0].set_data(list(range(len(accuracy_run))), accuracy_run)
                ax.figure.canvas.draw()

            if iteration > 1:
                h = ax.plot(list(range(len(accuracy_run))), accuracy_run, 'ro-')
            err = np.zeros(y.shape[0])

            accepted = False
            for i in range(bootstrap):
                model = reg.XGBoost(booster='gbtree', max_depth=4, early_stopping_rounds=10)
                acc = model.fit(x, ye, test_split=test_split)
                accuracy_run.append(acc)
                y_predicted = np.clip(model.predict(x), a_min=-1. + eps, a_max=1. - eps)
                err = err + (rev_transform(y_predicted) - y_actual)
                if (score is None) or (score > acc):
                    score = acc
                    best_model = model
                    accepted = True
                    print("Improved: Epoch: %d, BootStrap: %d, Model accuracy: %.5f" % (iteration, i, acc))
                else:
                    print("Maintained: Epoch: %d, Best accuracy: %.5f, Model accuracy: %.5f" % (iteration, score, acc))
            iteration = iteration + 1
            counter = counter + 1
            if accepted:
                counter = 0
            if len(accuracy_run) > run_avg:
                if acc_avg is None:
                    acc_avg = np.mean(accuracy_run[-run_avg:])
                if np.mean(accuracy_run[-run_avg:]) < acc_avg:
                    counter = 0
                    acc_avg = np.mean(accuracy_run[-run_avg:])

        model_out = os.path.join(os.path.dirname(__file__),
                                 'out',
                                 'revised_%s_%d_regression.dat' % (mol_type, max_dev))
        best_model.save(model_out)

        accuracy_out = os.path.join(os.path.dirname(__file__),
                                    'out',
                                    'revised_%s_%d_accuracy.csv' % (mol_type, max_dev))
        np.savetxt(accuracy_out, accuracy_run, delimiter=',')
        plt.close(fig)


