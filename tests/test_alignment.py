import os
from structural_dynamics import read_trajectory_catrace, align_trajectories

if __name__ == "__main__":
    trj_file1 = os.path.join(os.path.dirname(__file__), 'data', 'ex_trajectory_1.pdb')
    trj_file2 = os.path.join(os.path.dirname(__file__), 'data', 'ex_trajectory_2.pdb')
    assert os.path.isfile(trj_file1) and os.path.isfile(trj_file2)
    trj1 = [pair['A'] for pair in read_trajectory_catrace(trj_file1)]
    trj2 = [pair['A'] for pair in read_trajectory_catrace(trj_file2)]
    m, score, alignment = align_trajectories(trj1, trj2)
    print(alignment)
    print("Number of aligned state (%d) with average score (%.3f)" % (m, score))
