import os
import logging
import warnings
import argparse
import numpy as np
import coarse_graining as cg
import structural_dynamics as sd

__version__ = "1.0"


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
    logger = logging.getLogger('main.get_signatures')
    try:
        sig = cg.get_atom_position_signature(pdb, amino, properties=property_list, topn=nbrs)
        for aa in sig.keys():
            sig[aa] = np.array(sig[aa], dtype=np.float)
    except:
        logger.warning('Failed to process file [%s]' % pdbfile)
        sig = {}
    return sig


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
                        help="name of the output directory",
                        type=str, required=True)

    args = parser.parse_args()

    if not os.path.isdir(args.pdb_dir):
        raise Exception("Invalid pdb directory location [%s]" % args.pdb_dir)

    if not os.path.isdir(args.output):
        raise Exception("Output directory does not exists [%s]" % os.path.dirname(args.output))

    if os.path.isfile(args.output):
        warnings.warn("output file will be overwritten [%s]" % args.output)

    amino = sd.get_amino(args.amino.upper())
    properties = args.properties if len(args.properties) > 0 else cg.CoarseGrainProperty.default_properties()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('prepare_volume_feature_vectors')
    pdb_files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]

    all_features = {aa: None for aa in amino.atom_names() if aa != "CA"}
    for i, filename in enumerate(pdb_files):
        logging.info("Processing file [%s]" % filename)
        sig = get_signatures(filename,
                             amino,
                             property_list=properties,
                             nbrs=args.neighbors)
        if len(sig) == 0:
            continue
        for aa in sig.keys():
            if aa in all_features:
                if all_features[aa] is None:
                    all_features[aa] = sig[aa]
                else:
                    if all_features[aa].shape[1] == sig[aa].shape[1]:
                        all_features[aa] = np.concatenate((all_features[aa], sig[aa]), axis=0)
                    logger.info("Number of entries for [%s] in the feature definition (%d)" % (aa,
                                                                                               all_features[aa].shape[0]))

    for aa in all_features.keys():
        if all_features[aa] is not None:
            logger.info("final data size for %s : (%dx%d)" % (aa, *all_features[aa].shape))
            outfile = os.path.join(args.output, '%s_%s_position_features.csv' %
                                   (amino.name(one_letter_code=False).lower(), aa))
            np.savetxt(outfile, all_features, delimiter=',', fmt='%.5f')
