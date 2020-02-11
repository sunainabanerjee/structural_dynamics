import os
import logging
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, complete, linkage
from meta_learner import MetaLearnerXGBoost, ensemble_analyzer


if __name__ == "__main__":
    mol_type = 'atp'
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             'revised_%s_site_signature_without_tag.csv' % mol_type)
    logging.basicConfig(level=logging.INFO)
    assert os.path.isfile(data_file)
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape
    best_data, score, best_model = None, None, None
    out_dir = os.path.join(os.path.dirname(__file__), 'ensemble_models')
    x = data[:, :-1].reshape((n, f-1))
    y = data[:, -1].reshape(n)
    learner = MetaLearnerXGBoost(n_ensemble=5)
    learner.load(out_dir)
    # learner.minimum_datasize_factor(f=3.5)
    # learner.fit(x, y)
    distances = ensemble_analyzer(learner,
                                  sample_size=900,
                                  method='setdiff_1')
    print(distances)
    plt.figure()
    dn = dendrogram(complete(distances))
    plt.show()
    learner.save(out_dir)


