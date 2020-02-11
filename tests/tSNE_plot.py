import os
import re
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    feature_file = "/home/sumanta/PycharmProjects/" \
                   "structural_dynamics/scripts/" \
                   "site_signature/atp_site_signature_with_tag.csv"
    assert os.path.isfile(feature_file)
    data = pd.read_csv(feature_file, sep=',', header=None)
    col_names = list(data.columns)
    idx = [True if re.match('.*_e.*', s) else False for s in data[col_names[0]]]
    data = data.loc[idx, col_names]
    data_val = data[col_names[1:-1]].values
    data_tag = data[col_names[-1]].values
    x = TSNE(n_components=2,
             perplexity=10,
             learning_rate=500.,
             n_iter=5000,
             verbose=1).fit_transform(data_val)
    colors = ['r', 'g', 'b']
    x_tag = [colors[i+1] for i in data_tag]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(x[:, 0], x[:, 1], c=x_tag)
    plt.show()

