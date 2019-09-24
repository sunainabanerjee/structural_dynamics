import numpy as np
from geometry import *
from structural_dynamics import *
from sklearn.neighbors import NearestNeighbors
from .cg_properties import CoarseGrainProperty as CG
from .cg_properties import PropertyCalculator as PC

__version__ = "1.0"
__all__ = ['find_k_neighbors', 'find_k_neighbors_fast', 'cg_neighbor_signature']


def find_k_neighbors_fast(pdb, residues, k):
    if isinstance(pdb, PDBStructure):
        ca_trace = pdb_to_catrace(pdb)
    else:
        ca_trace = pdb
    assert isinstance(ca_trace, CaTrace) and len(ca_trace) > k
    assert isinstance(residues, list) and k > 0
    all_residues = ca_trace.residue_ids
    x = np.array([[*ca_trace.xyz(r)] for r in all_residues])
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="ball_tree").fit(x)
    nbr_list = {}
    y = np.array([[*ca_trace.xyz(r)] for r in residues])
    distances, indices = nbrs.kneighbors(y)
    for i, r in enumerate(residues):
        nbr_list[r] = [(all_residues[indices[i, j]],
                        distances[i, j]) for j in range(1, k+1)]
    return nbr_list


def find_k_neighbors(pdb, residues, k):
    if isinstance(pdb, PDBStructure):
        ca_trace = pdb_to_catrace(pdb)
    else:
        ca_trace = pdb
    assert isinstance(ca_trace, CaTrace) and len(ca_trace) > k
    assert isinstance(residues, list) and k > 0
    if k > 2:
        return find_k_neighbors_fast(ca_trace, residues, k)
    all_residues = ca_trace.residue_ids
    for r in residues:
        assert r in all_residues
    coordinates = {r: Coordinate3d(*ca_trace.xyz(r)) for r in all_residues}
    n = len(all_residues)
    neighbor_distances, nbrs = dict(), dict()
    for ri in residues:
        neighbor_distances[ri] = {'distance': list(), 'resid': list()}
        for rj in all_residues:
            if ri != rj:
                d = distance(coordinates[ri], coordinates[rj])
                neighbor_distances[ri]['distance'].append(d)
                neighbor_distances[ri]['resid'].append(rj)
        nbrs[ri] = [(neighbor_distances[ri]['resid'][idx],
                     neighbor_distances[ri]['distance'][idx])
                    for idx in np.argsort(neighbor_distances[ri]['distance'])[:k]]
    return nbrs


def cg_neighbor_signature(pdb,
                          residues,
                          properties=CG.default_properties(),
                          topn=12):
    if isinstance(pdb, PDBStructure):
        ca_trace = pdb_to_catrace(pdb)
    else:
        ca_trace = pdb
    assert isinstance(ca_trace, CaTrace) and len(ca_trace) > topn
    assert isinstance(residues, list)
    assert isinstance(properties, list)
    assert topn > 0
    for p in properties:
        assert p in CG.supported_properties()
    all_residues = ca_trace.residue_ids
    for r in residues:
        assert r in all_residues
    nbr_list = find_k_neighbors(pdb=ca_trace,
                                residues=residues,
                                k=topn)
    all_signatures = list()
    for r in residues:
        sig = list()
        if CG.bend() in properties:
            sig.append(PC.bend(ca_trace, r))
        idx = all_residues.index(r)
        assert (idx >= 0) and (idx < len(all_residues))
        assert topn == len(nbr_list[r])
        for nbr, d in nbr_list[r]:
            props = dict()
            if CG.distance() in properties:
                props[CG.distance()] = d
            if CG.angular() in properties:
                props[CG.angular()] = PC.angular(ca_trace, r, nbr)
            if CG.sasa() in properties:
                props[CG.sasa()] = PC.sasa(ca_trace, nbr)
            if CG.hydropathy() in properties:
                props[CG.hydropathy()] = PC.hydropathy(ca_trace, nbr)
            if CG.flexibility() in properties:
                props[CG.flexibility()] = PC.flexibility(ca_trace, nbr)
            if CG.volume() in properties:
                props[CG.volume()] = PC.volume(ca_trace, nbr)
            if CG.bend() in properties:
                props[CG.bend()] = PC.bend(ca_trace, nbr)
            for p in properties:
                sig.append(props[p])
        all_signatures.append(sig)
    return all_signatures

