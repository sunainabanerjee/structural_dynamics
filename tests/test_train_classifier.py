import os
import numpy as np
import classifier as cls


if __name__ == "__main__":
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'rna_signature.csv')
    assert os.path.isfile(data_file)
    with open(data_file, "r+") as fp:
        data = np.loadtxt(fp, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape
    x = data[:, :-1].reshape((n, f-1))
    y = data[:, -1]
    model = cls.XGBoost(booster=cls.XGBooster.gblinear(),
                        max_depth=4,
                        n_estimators=300,
                        n_class=3)
    acc = model.fit(x, y, test_split=0.2)
    print(acc)
    model_out = os.path.join(os.path.dirname(__file__), 'out', 'classify.dat')
    model.save(model_out)
    #print(model.loading_factors)



