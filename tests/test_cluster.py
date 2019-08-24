import os
import logging
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from structural_dynamics import read_trajectory_catrace, cluster_catraces

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    trj_file1 = os.path.join(os.path.dirname(__file__), 'data', 'modes1', 'mode_catraj_7.pdb')
    trj_file2 = os.path.join(os.path.dirname(__file__), 'data', 'modes1', 'mode_catraj_8.pdb')
    trj_file3 = os.path.join(os.path.dirname(__file__), 'data', 'modes1', 'mode_catraj_9.pdb')
    trj_file4 = os.path.join(os.path.dirname(__file__), 'data', 'modes1', 'mode_catraj_10.pdb')
    trj_file5 = os.path.join(os.path.dirname(__file__), 'data', 'modes1', 'mode_catraj_11.pdb')
    for file in (trj_file1, trj_file2, trj_file3, trj_file4, trj_file5):
        assert os.path.isfile(file)
    trj1 = [pair['A'] for pair in read_trajectory_catrace(trj_file1)]
    trj2 = [pair['A'] for pair in read_trajectory_catrace(trj_file2)]
    trj3 = [pair['A'] for pair in read_trajectory_catrace(trj_file3)]
    trj4 = [pair['A'] for pair in read_trajectory_catrace(trj_file4)]
    trj5 = [pair['A'] for pair in read_trajectory_catrace(trj_file5)]
    all_trj = trj1[::3] + trj2[::3] + trj3[::3] + trj4[::3] + trj5[::3]
    cls = cluster_catraces(all_trj)
    dendrogram(cls,
               color_threshold=1,
               show_leaf_counts=True)
    plt.show()

