import os
import time
import logging
from geometry import *
from mutant_model import *
from structural_dynamics import *


if __name__ == "__main__":
    logging.basicConfig(debug=logging.DEBUG)
    ref_pdb_file = os.path.join(os.path.dirname(__file__), 'data', 'structure_1a1v.pdb')
    trj_dir = os.path.join(os.path.dirname(__file__), 'data', 'modes1')

    assert os.path.isfile(ref_pdb_file)
    ref_trace = pdb_to_catrace(read_pdb(pdb_file=ref_pdb_file)[0]['A'])
    max_crd, min_crd = Coordinate3d(28.0, 54.0, 32.0), Coordinate3d(4.0, 11.0, 18.0)
    for i in range(len(max_crd)):
        max_crd[i] += 2
        min_crd[i] -= 2

    assert os.path.isdir(trj_dir)
    trajectories = []
    for f in os.listdir(trj_dir):
        if f.endswith(".pdb") and f.startswith("mode_catraj"):
            trj_file = os.path.join(trj_dir, f)
            trj = [pair['A'] for pair in read_trajectory_catrace(trj_file)]
            trajectories.append(trj)

    start = time.time()
    trj_handler = NMATrajectoryHandler(min_coordinate=min_crd, max_coordinate=max_crd)
    for trj in trajectories:
        trj = align_trajectory(ref_trace, trj[0], trj, ca_trace=True)
        trj_handler.add_trajectory(trj)
    end = time.time()
    print("Total processing time: %.2f" % (end - start))
    sasa_model_folder = "/home/sumanta/PycharmProjects/structural_dynamics/mutant_model/models/sasa_models"
    vol_model_folder = "/home/sumanta/PycharmProjects/structural_dynamics/mutant_model/models/volume_models"
    assert os.path.isdir(sasa_model_folder)
    assert os.path.isdir(vol_model_folder)
    start = time.time()
    sasa_model = BulkSasaRegressor(model_folder=sasa_model_folder)
    vol_model = BulkVolumeRegressor(model_folder=vol_model_folder)
    end = time.time()
    print("Model load time %.5f" % (end - start))

    start=time.time()
    sasa_sig = sasa_model.signature(trj_handler)
    end = time.time()
    print("Sasa Prediction time: %.3f" % (end - start))

    start=time.time()
    vol_sig, occ_sig = vol_model.signature(trj_handler)
    end = time.time()
    print("Volume Prediction time: %.3f" % (end - start))



