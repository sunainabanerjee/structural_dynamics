import os
import numpy as np
import classifier as cls
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

__version__ = "1.0"


if __name__ == "__main__":
    mol_type = 'rna'
    data_file = os.path.join(os.path.dirname(__file__),
                             '..',
                             'scripts',
                             'site_signature',
                             '%s_feature_vector_without_tag_jan2020.csv' % mol_type)
    assert os.path.isfile(data_file)
    data = np.loadtxt(data_file, delimiter=',')
    np.random.shuffle(data)
    n, f = data.shape
    iteration, test_split = 15, 0.4
    best_data, score, best_model = None, None, None

    x = data[:, :-1].reshape((n, f - 1))
    y = data[:, -1]
    best_roc = None

    for i in range(iteration):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split)
        model = cls.XGBoost(n_class=3,
                            booster='gbtree',
                            n_estimators=200,
                            learning_rate=0.1,
                            subsample=0.9,
                            max_depth=10)
        acc = model.fit(x_train, y_train, test_split=0.1)
        x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=test_split)
        sample_mean = model.predict_proba(x)
        roc = cls.evaluate_roc(y_test,
                               cls.bias_correction(sample_mean,
                                                   model.predict_proba(x_test)),
                               report_combined=True,
                               classes=model.out_class_names,
                               n=50)
        roc_area = cls.auc(roc['fpr'].tolist(), roc['tpr'].tolist())
        if (score is None) or (score < roc_area):
            score = roc_area
            best_model = model
            best_roc = roc.copy()
            print("AUC (ROC): %.5f , Iteration: %d" % (roc_area, i+1))
        else:
            print("AUC (ROC): %.5f, Best AUC (ROC): %.5f" % (roc_area, score))

    model_out = os.path.join(os.path.dirname(__file__),
                             'out',
                             'auc_%s_classify.dat' % mol_type)
    best_model.save(model_out)

    plt.plot(np.linspace(0, 1, 20),
             np.linspace(0, 1, 20),
             color='#B09C85FF',
             linestyle='dashed')
    plt.plot(best_roc['fpr'], best_roc['tpr'], color='#8491B4FF')
    plt.text(0.75, 0.15, "AUC: %.3f" % cls.auc(best_roc['fpr'].tolist(), best_roc['tpr'].tolist()), fontsize=14)
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.xlim((-0.01, 1.01))
    plt.ylim((-0.01, 1.01))
    plt.tick_params(axis="both", labelsize='large')

    plt.grid('off')
    plt.show()



