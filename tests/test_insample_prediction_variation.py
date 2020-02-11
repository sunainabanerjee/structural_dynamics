import os
import numpy as np
import pandas as pd
import seaborn as sns
from classifier import XGBoost
from matplotlib import pyplot as plt

if __name__ == "__main__":
    mol_type = 'rna'
    model_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'site_signature')
    assert os.path.isdir(model_folder) and os.path.isdir(data_folder)

    model_file = os.path.join(model_folder,
                              'revised_%s_classify.dat' % mol_type)

    data_file = os.path.join(data_folder,
                             '%s_feature_vector_with_tag_jan2020.csv' % mol_type)

    assert os.path.isfile(model_file) and os.path.isfile(data_file)

    data = pd.read_csv(data_file, header=None, sep=',')

    n_data, ncol = data.shape
    mutant_tags = np.array([s.split("_")[0] for s in data[0]])
    mutant_type = np.array([s.split("_")[2][0] if len(s.split("_")) == 4 else 'c' for s in data[0]])

    indices = list(range(1, ncol-1))
    x = data[indices].values
    mutant_class = np.array(data[ncol-1].tolist(), dtype=np.float)

    cls = XGBoost(n_class=3)
    cls.load(model_file)
    p = cls.predict_proba(x)

    class_order = cls.out_class_names
    mutants = np.unique(mutant_tags)
    predicted_class = np.array([ class_order[i] for i in np.argmax(p, axis=1)])

    for m in mutants:
        correctly_predicted_inst = np.where((predicted_class == mutant_class) & (mutant_tags == m))[0]
        relevant_instances = np.where(mutant_tags == m)[0]
        av_acc = len(correctly_predicted_inst)/len(relevant_instances)

        correctly_predicted_inst = np.where((predicted_class == mutant_class) &
                                            (mutant_tags == m) &
                                            (mutant_type == 'd'))[0]
        relevant_instances = np.where((mutant_tags == m) &
                                      (mutant_type == 'd'))[0]
        dyn_acc = len(correctly_predicted_inst)/len(relevant_instances)

        correctly_predicted_inst = np.where((predicted_class == mutant_class) &
                                            (mutant_tags == m) &
                                            (mutant_type == 'e'))[0]
        relevant_instances = np.where((mutant_tags == m) &
                                      (mutant_type == 'e'))[0]
        eq_acc = len(correctly_predicted_inst)/len(relevant_instances)

        actual_tag = np.unique(mutant_class[np.where(mutant_tags == m)[0]])[0]
        print("%s (%d): %5.2f %5.2f %5.2f" % (m, actual_tag, av_acc, eq_acc, dyn_acc))

    pred = dict()
    for i, cls in enumerate(class_order):
        cls_idx = np.where(mutant_class == cls)[0]
        p_right = p[cls_idx, i]
        best_predict = np.argmax(p[cls_idx, :], axis=1)
        correct_idx = np.where(best_predict == i)[0]
        p_wrong = p[cls_idx, best_predict]
        p_wrong[correct_idx] = 0
        pred[cls] = {'correct': p_right, 'incorrect': p_wrong}

    colors = ['#DC0000FF', '#B09C85FF', '#8491B4FF']
    legend = ['Deficient', 'Basal', 'Enhanced']
    fig, axs = plt.subplots(1, 1, sharex='row')
    for i, cls in enumerate(class_order):
        sns.distplot(pred[class_order[i]]['correct'],
                     ax=axs,
                     hist=False,
                     kde=True,
                     bins=100,
                     kde_kws={
                              "kernel": 'cos',
                              "bw": 0.05
                             },
                     color=colors[i],
                     label=legend[i])

    sns.set_context("paper", font_scale=1.5)
    axs.set_xlabel('Probability of correct assignment', size=16)
    axs.set_ylabel('Normalized frequency', size=16)

    sns.set_style('whitegrid', {'font.family': 'serif', 'font.serif': 'Arial', 'xtick.labelsize': 32, 'ytick.labelsize': 16})
    axs.legend(loc="upper left")
    axs.title.set_text("%s Model Accuracy" % mol_type.upper())
    plt.xlim((0.0, 1.05))
    plt.grid(False)
    plt.show()

