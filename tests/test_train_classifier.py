import os
import numpy as np
import classifier as cls


if __name__ == "__main__":
    mol_type = 'atp'
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_feature_vector_without_tag_jan2020.csv' % mol_type)
    assert os.path.isfile(data_file)
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape
    iteration = 20
    best_data, score, best_model = None, None, None
    for i in range(iteration):
        x = data[:, :-1].reshape((n, f-1))
        y = data[:, -1]
        model = cls.XGBoost(n_class=3)
        # model.set_pthresh(0.25)
        acc = model.fit(x, y, test_split=0.2)
        if (score is None) or (score < acc):
            score = acc
            best_model = model
            print("Accuracy: %.2f , Iteration: %d" % (acc*100, i+1))
        else:
            print("Accuracy: %.2f, Best Accuracy: %.2f" % (acc*100, score*100))
    model_out = os.path.join(os.path.dirname(__file__),
                             'out',
                             'revised_%s_classify.dat' % mol_type)
    best_model.save(model_out)



