import numpy as np
from geometry import *
from structural_dynamics import *

__version__ = "1.0"
__all__ = ['align_nma_trajectories',
           'aligning_rotation',
           'align_trajectory',
           'reset_pdb_coordinate']


def align_nma_trajectories(trj1, trj2, return_array=False):
    assert isinstance(trj1, list) and isinstance(trj2, list)
    assert (len(trj1) == len(trj2)) and (len(trj1) > 1)
    assert isinstance(trj1[0], CaTrace) and isinstance(trj2[0], CaTrace)

    trj1_arr = [snap.coordinate_array() for snap in trj1]
    trj2_arr = [snap.coordinate_array() for snap in trj2]
    rev_trj2 = [trj2_arr[0]] + trj2_arr[1:][::-1]

    n = len(trj1)
    forward_score = [kabsch_rmsd(trj1_arr[i], trj2_arr[i], return_matrix=False, return_coordinate=False) for i in range(n)]
    reverse_score = [kabsch_rmsd(trj1_arr[i], rev_trj2[i], return_matrix=False, return_coordinate=False) for i in range(n)]
    if np.sum(forward_score) < np.sum(reverse_score):
        forward_match = True
        score = np.sum(forward_score)
        aligned_index = list(range(n))
    else:
        forward_match = False
        score = np.sum(reverse_score)
        aligned_index = [0] + list(reversed(range(1, n)))

    if return_array:
        return trj1_arr, [trj2_arr[i] for i in aligned_index], score, forward_match
    else:
        return trj1, [trj2[i] for i in aligned_index], score, forward_match


def aligning_rotation(ref_structure,
                      tgt_structure,
                      align_pair=None):
    assert isinstance(ref_structure, CaTrace)
    assert isinstance(tgt_structure, CaTrace)
    all_residue_ref = ref_structure.residue_ids
    all_residue_tgt = tgt_structure.residue_ids
    if align_pair is None:
        assert len(ref_structure) == len(tgt_structure)
        align_pair = [(all_residue_ref[i], all_residue_tgt[i]) for i in range(len(all_residue_ref))]
    assert isinstance(align_pair, list) and len(align_pair) <= min(len(tgt_structure), len(ref_structure))
    assert len(align_pair) > 3
    for r1, r2 in align_pair:
        assert r1 in all_residue_ref
        assert r2 in all_residue_tgt
    arr_ref = ref_structure.coordinate_array(residue_list=[r1 for r1, r2 in align_pair])
    arr_tgt = tgt_structure.coordinate_array(residue_list=[r2 for r1, r2 in align_pair])
    r, m = kabsch_rmsd(arr_tgt, arr_ref, return_matrix=True, return_coordinate=False)
    return m


def reset_pdb_coordinate(snapshot, coordinates):
    assert isinstance(snapshot, CaTrace)
    if isinstance(coordinates, list):
        coordinates = np.array([[row[0], row[1], row[2]] for row in coordinates], dtype=np.float)
    assert isinstance(coordinates, np.ndarray)
    assert len(snapshot) == coordinates.shape[0]
    all_residues = snapshot.residue_ids
    entries = [{'resid': all_residues[i],
                'resname': snapshot.get_amino(all_residues[i]).name(one_letter_code=False),
                'bfactor': snapshot.b_factor(all_residues[i]),
                'x': coordinates[i, 0],
                'y': coordinates[i, 1],
                'z': coordinates[i, 2]} for i in range(coordinates.shape[0])]
    return CaTrace(name=snapshot.name,
                   entry=entries,
                   chainId=snapshot.chain)


def align_trajectory(ref_structure,
                     tgt_structure,
                     trajectory,
                     align_pair=None,
                     ca_trace=False):
    assert isinstance(trajectory, list) and len(trajectory) > 0
    m = aligning_rotation(ref_structure, tgt_structure, align_pair=align_pair)
    aligned_trj = []
    for snapshot in trajectory:
        assert isinstance(snapshot, CaTrace)
        centroid = np.mean(ref_structure.coordinate_array(), axis=0)
        coordinate_array = snapshot.coordinate_array()
        coordinate_array = coordinate_array - np.mean(coordinate_array, axis=0)
        coordinate_array = np.matmul(coordinate_array, m) + centroid
        aligned_trj.append(coordinate_array)
    if ca_trace:
        return [reset_pdb_coordinate(trajectory[i], aligned_trj[i]) for i, crd in enumerate(aligned_trj)]
    else:
        return aligned_trj

