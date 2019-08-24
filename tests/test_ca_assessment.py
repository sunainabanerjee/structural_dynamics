import os
import logging
from structural_dynamics import *

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    wild_trj_file = os.path.join(os.path.dirname(__file__), 'data', 'wild_good_traj.pdb')
    mutn_trj_file = os.path.join(os.path.dirname(__file__), 'data', 'mutant_good_traj.pdb')
    wild_ca_traj = [pair['A'] for pair in read_trajectory_catrace(wild_trj_file)]
    mutant_ca_traj = [pair['A'] for pair in read_trajectory_catrace(mutn_trj_file)]
    wild_fix_traj = [fix_ca_trace(traj) for traj in wild_ca_traj]
    mutant_fix_traj = [fix_ca_trace(traj) for traj in mutant_ca_traj]
    out_file = os.path.join(os.path.dirname(__file__), 'out', 'wild_fixed.pdb')
    with open(out_file, 'w+') as fp:
        for i, traj in enumerate(wild_fix_traj):
            fp.write("MODEL    %5d\n" % (i+1))
            traj.write(fp)
            fp.write("ENDMDL\n")
    out_file = os.path.join(os.path.dirname(__file__), 'out', 'mutant_fixed.pdb')
    with open(out_file, 'w+') as fp:
        for i, traj in enumerate(mutant_fix_traj):
            fp.write("MODEL    %5d\n" % (i+1))
            traj.write(fp)
            fp.write("ENDMDL\n")
