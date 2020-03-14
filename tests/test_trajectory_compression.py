import os
from structural_dynamics import *


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ex_trajectory = os.path.join(curr_dir, 'data', 'modes1', 'mode_catraj_7.pdb')
    assert os.path.isfile(ex_trajectory)
    trajectory = read_trajectory_catrace(ex_trajectory)
    trajectory = [snp[list(snp.keys())[0]]  for snp in trajectory]
    compressed_file = os.path.join(curr_dir, 'data', 'modes1', 'mode_catraj_7.cmpr')
    CompressedTrajectoryStore.save(trajectory, compressed_file)
    uncompress = CompressedTrajectoryStore.load(compressed_file)
    

