import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    iteration = 20
    best_data, score, best_model = None, None, None
    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1]

    xf = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    xp = pca.fit_transform(x)
    nx = np.concatenate((xp, y.reshape((xp.shape[0], 1))), axis=1)
    df = pd.DataFrame(nx, columns=['p1', 'p2', 'y'])
    df.to_csv('/tmp/t.csv', index=False, header=True, sep=',')




