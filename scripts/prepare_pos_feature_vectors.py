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
    sig = None
    logger = logging.getLogger('main.get_signatures')
    try:
        resids = pdb.find_residue(amino.name(one_letter_code=True))
        if len(resids) > 0:
            resids, sig, diheds = cg.get_model_dihedral_signature(pdb,
                                                                  resids,
                                                                  properties=property_list,
                                                                  topn=nbrs)
            assert len(sig) == len(diheds)
            sig = np.concatenate([np.array(sig, dtype=np.float), np.array(diheds, dtype=np.float)], axis=1)
    except:
        logger.warning('Failed to process file [%s]' % pdbfile)
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
                        help="name of the output file",
                        type=str, required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('prepare_volume_feature_vectors')

    if not os.path.isdir(args.pdb_dir):
        raise Exception("Invalid pdb directory location [%s]" % args.pdb_dir)

    if not os.path.isdir(os.path.dirname(os.path.abspath(args.output))):
        raise Exception("Output directory does not exists [%s]" % os.path.dirname(args.output))

    if os.path.isfile(args.output):
        logger.warning("File will be overwritten [%s]" % args.output)

    amino = sd.get_amino(args.amino.upper())
    properties = args.properties if len(args.properties) > 0 else cg.CoarseGrainProperty.default_properties()

    pdb_files = [os.path.join(args.pdb_dir, f) for f in os.listdir(args.pdb_dir) if f.endswith('.pdb')]
    all_sig = None

    for i, filename in enumerate(pdb_files):
        logging.info("Processing file [%s]" % filename)
        sig = get_signatures(filename,
                             amino,
                             property_list=properties,
                             nbrs=args.neighbors)
        if all_sig is None:
            all_sig = sig
        elif sig is not None:
            all_sig = np.concatenate([all_sig, sig], axis=0)
            print("Feature Size: %d x %d" % all_sig.shape)

    np.savetxt(args.output, all_sig, delimiter=',', fmt="%.5f")
