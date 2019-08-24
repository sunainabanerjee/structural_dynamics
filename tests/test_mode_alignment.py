import os
import logging
from structural_dynamics import read_trajectory_catrace, mode_alignment

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    trj_file1 = os.path.join(os.path.dirname(__file__), 'data', 'modes1')
    trj_file2 = os.path.join(os.path.dirname(__file__), 'data', 'modes2')
    assert os.path.isdir(trj_file1) and os.path.isdir(trj_file2)
    modes1 = [[pair['A'] for pair in read_trajectory_catrace(os.path.join(trj_file1, f))]
              for f in os.listdir(trj_file1) if f.endswith('.pdb')]
    modes2 = [[pair['A'] for pair in read_trajectory_catrace(os.path.join(trj_file2, f))]
              for f in os.listdir(trj_file2) if f.endswith('.pdb')]

    print(mode_alignment(modes1, modes2, min_score=4.5))
