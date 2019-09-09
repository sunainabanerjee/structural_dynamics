import logging
import numpy as np
from multiprocessing.pool import ThreadPool
from scipy.optimize import linear_sum_assignment
from structural_dynamics.pdb_processor import CaTrace
from geometry import Coordinate3d
from geometry import kabsch_coordinate

__version__ = "1.0"
__all__ = ['catrace_coordinates', 'catraces_rmsd',
           'trajectory_alignment_matrix', 'align_trajectories',
           'mode_alignment']


def catrace_coordinates(catrace,
                        order_list=None):
    assert isinstance(catrace, CaTrace)
    if order_list is None:
        order_list = catrace.residue_ids
    assert isinstance(order_list, list)
    return [Coordinate3d(*catrace.xyz(r)) for r in order_list]


def catraces_rmsd(catrace1,
                  catrace2,
                  order_list=None,
                  return_structure=False):
    assert isinstance(catrace1, CaTrace)
    assert isinstance(catrace2, CaTrace)
    all_structure = (order_list is None) or \
                    ((len(order_list) == len(catrace1)) and
                     (len(order_list) == len(catrace2)))
    if order_list is None:
        assert len(catrace1) == len(catrace2)
        r1 = catrace1.residue_ids
        r2 = catrace2.residue_ids
        order_list = [(x,y) for x, y in zip(r1, r2)]
    assert isinstance(order_list, list)
    for item in order_list:
        assert len(item) == 2
    r1 = [x for x, y in order_list]
    r2 = [y for x, y in order_list]
    if all_structure and return_structure:
        rms, coords = kabsch_coordinate(catrace_coordinates(catrace1, r1),
                                        catrace_coordinates(catrace2, r2),
                                        return_coordinate=return_structure)
        items = [{ 'resid': r,
                   'resname': catrace1.get_amino(r).name(one_letter_code=False),
                   'x': coords[i].x,
                   'y': coords[i].y,
                   'z': coords[i].z} for i, r in enumerate(catrace1.residue_ids)]
        return rms, CaTrace(catrace1.name, items, chainId=catrace1.chain)

    return kabsch_coordinate(catrace_coordinates(catrace1, r1),
                             catrace_coordinates(catrace2, r2),
                             return_coordinate=return_structure)


def trajectory_alignment_matrix(traj1,
                                traj2,
                                residue_map=None):
    assert isinstance(traj1, list) and isinstance(traj2, list)
    n1, n2 = len(traj1), len(traj2)
    assert (n1 > 0) and (n2 > 0)
    assert isinstance(traj1[0], CaTrace) and isinstance(traj2[0], CaTrace)
    if residue_map is None:
        r1 = traj1[0].residue_ids
        r2 = traj2[0].residue_ids
        assert len(r1) == len(r2)
        residue_map = [(x, y) for x, y in zip(r1, r2)]
    assert isinstance(residue_map, list)
    for i in range(len(residue_map)):
        assert len(residue_map[i]) == 2
    scores = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            scores[i, j] = catraces_rmsd(traj1[i],
                                         traj2[i],
                                         order_list=residue_map,
                                         return_structure=False)
    return scores


def alignment(score_matrix, epsilon=0.1):
    assert isinstance(score_matrix, np.ndarray )
    r, c = score_matrix.shape
    row_vector, tracking_matrix = np.zeros(c+1), np.zeros((r, c))
    tracking_score = np.zeros((r,c))
    penalty = score_matrix.min() - epsilon
    for i in range(r):
        nxt_row = np.zeros(c+1)
        nxt_row[0] = row_vector[0] + penalty
        for j in range(1,c+1):
            matc = row_vector[j-1] + score_matrix[i,j-1]
            inst = nxt_row[j-1] + penalty
            dele = row_vector[j] + penalty
            nxt_row[j] = max(matc, inst, dele)
            track = [0, 1, -1]
            idx = [matc, inst, dele].index(nxt_row[j])
            tracking_matrix[i, j-1] = track[idx]
            tracking_score[i, j-1] = nxt_row[j]
        row_vector = nxt_row.copy()
    p = row_vector.tolist().index(row_vector.max()) - 1
    match_seq = []
    x, y = r - 1, p
    while x >= 0:
        if tracking_matrix[x, y] == 0:
            match_seq.append((x, y))
            x = x - 1
            y = y - 1
        elif tracking_matrix[x, y] == 1:
            match_seq.append(('-', y))
            y = y - 1
        elif tracking_matrix[x, y] == -1:
            match_seq.append((x, '-'))
            x = x - 1
    return list(reversed(match_seq)), tracking_score


def align_trajectories(traj1,
                       traj2,
                       residue_map=None):
    score_matrix = trajectory_alignment_matrix(traj1,
                                               traj2,
                                               residue_map=residue_map)
    align_matrix = -1 * np.concatenate((score_matrix, score_matrix), axis=1)
    n = score_matrix.shape[1]
    aln_raw, score = alignment(align_matrix)
    aln = [(x, y % n) if y != '-' else (x, y) for x, y in aln_raw]
    match = np.sum([1 if (x != '-') and (y != '-') else 0 for x, y in aln])
    score = -1 * np.mean([align_matrix[x, y] for x, y in aln_raw if (x != "-") and (y != "-")])

    xrev_align_matrix = align_matrix[::-1]
    aln_raw, score_rev = alignment(xrev_align_matrix)
    m = score_rev.shape[0] - 1
    aln_rev = [(m - x, y % n) if y != '-' else (x, y) for x, y in aln_raw]
    match_rev = np.sum([1 if (x != '-') and (y != '-') else 0 for x, y in aln_rev])
    score_rev = -1 * np.mean([xrev_align_matrix[x, y] for x, y in aln_raw if (x != "-") and (y != "-")])

    if score_rev < score:
        return match_rev, score_rev, aln_rev
    else:
        return match, score, aln


def mode_alignment(mode1, mode2, residue_map=None, min_score=1.0):
    logger = logging.getLogger('structural_dynamics.mode_alignment')
    assert isinstance(mode1, list) and isinstance(mode2, list)
    scores = [[min_score for j in range(len(mode2))] for i in range(len(mode1))]
    hits = 0
    for i, traj1 in enumerate(mode1):
        assert isinstance(traj1, list)
        for j, traj2 in enumerate(mode2):
            assert isinstance(traj2, list)
            logger.debug('Calculating score between mode (%d, %d)' % (i, j))
            match, score, aln = align_trajectories(traj1,
                                                   traj2,
                                                   residue_map=residue_map)
            logger.debug('Alignment score (%.3f)' % score)
            if score < min_score:
                scores[i][j] = score
                hits += 1
    if hits > 0:
        row_ind, col_ind = linear_sum_assignment(scores)
        assert len(row_ind) == len(col_ind)
        return [(x, y) for x, y in zip(row_ind, col_ind) if scores[x][y] < min_score]
    else:
        return []



