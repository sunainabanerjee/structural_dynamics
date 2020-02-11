import os
import numpy as np
import classifier as cls
from matplotlib import pyplot as plt


if __name__ == "__main__":
    mol_type = 'atp'
    model_file = os.path.join(os.path.dirname(__file__),
                              'out',
                              'auc_%s_classify.dat' % mol_type)
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_feature_vector_without_tag_jan2020.csv' % mol_type)
    assert os.path.isfile(model_file)
    assert os.path.isfile(data_file)
    model = cls.XGBoost(n_class=3)
    model.load(model_file)
    classes = model.out_class_names
    data = np.loadtxt(data_file, delimiter=',')
    select_idx = np.random.choice(range(data.shape[0]), int(0.5*data.shape[0]))
    x = data[select_idx, :-1]
    y = data[select_idx, -1]
    y_proba = model.predict_proba(x)
    print(np.mean(y_proba, axis=0))
    roc = cls.evaluate_roc(y, y_proba, n=50)
    roc_c = cls.evaluate_roc(y, cls.bias_correction(y_proba), n=50)
    plt.plot(np.linspace(0, 1, 20),
             np.linspace(0, 1, 20),
             color='#B09C85FF',
             linestyle='dashed')
    plt.plot(roc['fpr'], roc['tpr'], color='#8491B4FF')
    plt.plot(roc_c['fpr'], roc_c['tpr'], color='red')
    plt.text(0.75, 0.15, "AUC: %.3f" % cls.auc(roc['fpr'].tolist(), roc['tpr'].tolist()), fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.tick_params(axis="both", labelsize='large')
    plt.grid('off')
    plt.show()


