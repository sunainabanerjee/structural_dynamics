import os
import argparse
import numpy as np
from geometry import *
import coarse_graining as cg
import structural_dynamics as sd

__version__ = "1.0"
__all__ = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot validate volume coarse grained "
                                                 "model prediction accuracy. Handles "
                                                 "only one chain at a time")

    parser.add_argument('--pdb-file', action='store', dest='pdb_file',
                        type=str, required=True,
                        help="target pdb file against which the prediction will be performed")

    parser.add_argument('--chain', action='store', dest='chain',
                        type=str, required=False, default='A',
                        help='chain of the target structure, default chain ID [A]')

    parser.add_argument('--model-folder', action='store', dest='model_folder',
                        type=str, required=True,
                        help="Model file containing the sasa model weights and definition")

    parser.add_argument('--model-type', action='store', dest='model_type',
                        type=str, choices=['mlp', 'xgb'], required=False,
                        default='mlp',
                        help="Sasa model type")

    parser.add_argument('--residue', nargs='+', dest='residues',
                        required=False, default=list(), type=int,
                        help="residues on which the prediction needs to be performed! "
                             "Default behaviour considers all residues")

    args = parser.parse_args()

    if not os.path.isfile(args.pdb_file):
        raise Exception("Not a valid PDB file [%s]" % args.pdb_file)

    if not os.path.isdir(args.model_folder):
        raise Exception("Not a valid model folder [%s]" % args.model_folder)

    pdb_list = sd.read_pdb(args.pdb_file)
    chain = args.chain
    residues = args.residues

    assert len(pdb_list) == 1

    vol_predictor = cg.VolumePredictor(model_folder=args.model_folder,
                                       model_type=args.model_type,
                                       min_occupancy=0.0)

    for pdbs in pdb_list:
        assert isinstance(pdbs, dict)
        assert chain in pdbs
        pdb_copy = pdbs[chain]
        ca_trace = sd.pdb_to_catrace(pdbs[chain])
        all_residues = ca_trace.residue_ids[1:-1]
        if len(residues) == 0:
            residues = all_residues
        assert all([r in all_residues for r in residues])
        recons_unit = vol_predictor.predict_volume(ca_trace,
                                                   residues)

        for resId, coordinates, occupancy in recons_unit:
            atoms = pdb_copy.atom_names(resId)
            a_crd = [Coordinate3d(*pdb_copy.xyz(resId, aa)) for aa in atoms]
            vdw_rad = [sd.get_vdw_radius(aa) for aa in atoms]
            assert len(a_crd) == len(vdw_rad)
            assert coordinates.shape[0] == len(occupancy)
            intersection_unit, union_unit = set(), set()
            intersection_volume, union_volume = list(), list()
            for i, occ in enumerate(occupancy):
                g_crd = Coordinate3d(coordinates[i, 0],
                                     coordinates[i, 1],
                                     coordinates[i, 2])
                g_rad = 2.36
                total_intersection = 0
                for j, crd in enumerate(a_crd):
                    vdw = vdw_rad[j]
                    vol = sphere_intersection_volume(g_rad, g_crd, vdw, crd)
                    total_intersection = total_intersection + vol
                    if (vol > 1) and (occ > 1):
                        intersection_unit.add(i)
                    if (vol > 1) or (occ > 1):
                        union_unit.add(i)
                intersection_volume.append(max(min(total_intersection, occ), 0))
                union_volume.append(max(total_intersection, occ, 0))
            print('%d\t%s\t%.5f\t%.5f' % (resId,
                                          pdb_copy.residue_name(residue_id=resId,
                                                                one_letter_code=False),
                                          len(intersection_unit)/len(union_unit),
                                          np.sum(intersection_volume)/np.sum(union_volume)))



