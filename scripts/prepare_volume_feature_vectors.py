import os
import logging
import warnings
import argparse
import numpy as np
import coarse_graining as cg
import structural_dynamics as sd


def get_signatures(pdbfile,
                   amino,
                   property_list,
                   nbrs):
    assert os.path.isfile(pdbfile)
    assert isinstance(property_list, list)
    assert nbrs > 2
    assert all([p in cg.CoarseGrainProperty.supported_properties() for p in property_list])
    pdblist = sd.read_pdb(pdbfile)
    assert len(pdblist) > 0
    chain = list(pdblist[0].keys())[0]
    pdb = pdblist[0][chain]
    try:
        ca_trace = sd.pdb_to_catrace(pdb)
    except:
        return np.array([])
    assert isinstance(pdb, sd.PDBStructure)
    residue_ids = [r for r in pdb.residue_ids if r > 0][1:-1]
    residue_ids = [r for r in residue_ids if sd.get_amino(pdb.residue_name(r)) == amino]
    residue_ids = cg.filter_complete_residues(pdb, residue_ids)
    vol = cg.Volume3D(size=cg.VolumeSignatureGrid.size(),
                      dim=cg.VolumeSignatureGrid.dim())
    if len(residue_ids) > 0:
        vol_signature = []
        for r in residue_ids:
            vol_signature.append(vol.get_signature(pdb, r, reset=True))
        nbr_signature = cg.cg_neighbor_signature(ca_trace,
                                                 residues=residue_ids,
                                                 properties=property_list,
                                                 topn=nbrs)
        vol_signature, nbr_signature = np.array(vol_signature), np.array(nbr_signature)
        if vol_signature.shape[0] == nbr_signature.shape[0]:
            return np.concatenate((nbr_signature, vol_signature), axis=1)
    return np.array([])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility builds the input for sasa prediction")

    parser.add_argument('--amino', action='store', dest='amino',
                        help="one/three letter amino acid code (case insensitive)",
                        type=str, required=True)

    parser.add_argument('--pdb-dir', action='store', dest='pdb_dir',
                        help="directory containing all the corresponding pdb files",
                        type=str, required=True)

    parser.add_argument('--neighbors', action='store', dest='neighbors',
                        default=12,
                        help="number of residue neighbors to evaluate",
                        type=int, required=False)

    parser.add_argument('--property', action='append', dest='properties',
                        default=[],
                        choices=cg.CoarseGrainProperty.supported_properties(),
                        help="properties to use for feature vector building", required=False)

    parser.add_argument('--output', action='store', dest='output',
                        help="name of the output file",
                        type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.pdb_dir):
        raise Exception("Invalid pdb directory location [%s]" % args.pdb_dir)

    if not os.path.isdir(os.path.dirname(args.output)):
        raise Exception("Output directory does not exists [%s]" % os.path.dirname(args.output))

    if os.path.isfile(args.output):
        warnings.warn("output file will be overwritten [%s]" % args.output)

    amino = sd.get_amino(args.amino.upper())
    properties = args.properties if len(args.properties) > 0 else cg.CoarseGrainProperty.default_properties()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('prepare_volume_feature_vectors')
    pdb_files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]

    all_features = None
    for i, filename in enumerate(pdb_files):
        sig = get_signatures(filename,
                             amino,
                             property_list=properties,
                             nbrs=args.neighbors)
        if sig.shape[0] == 0:
            continue
        if all_features is None:
            all_features = sig
        else:
            if all_features.shape[1] == sig.shape[1]:
                all_features = np.concatenate((all_features, sig), axis=0)
                logger.info("Number of entries in the feature definition (%d)" % all_features.shape[0])
    if all_features is not None:
        logger.info("final data size: (%dx%d)" % all_features.shape)
        np.savetxt(args.output, all_features, delimiter=',', fmt='%.5f')



