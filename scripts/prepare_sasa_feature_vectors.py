import os
import logging
import warnings
import argparse
import numpy as np
import coarse_graining as cg
import structural_dynamics as sd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility builds the input for sasa prediction")
    parser.add_argument('--json-dir', action='store', dest='json_dir',
                        help="Directory containing freesasa json outputs",
                        type=str, required=True)

    parser.add_argument('--amino', action='store', dest='amino',
                        help="one/three letter amino acid code (case insensitive)",
                        type=str, required=True)

    parser.add_argument('--pdb-dir', action='store', dest='pdb_dir',
                        help="directory containing all the corresponding pdb files",
                        type=str, required=True)

    parser.add_argument('--area-type', action='store', dest='area_type',
                        choices=cg.SASAJsonFields().area_fields,
                        default=cg.SASAJsonFields().total_area,
                        help="select the type of area needs to be parsed",
                        type=str, required=False)

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
    if not os.path.isdir(args.json_dir):
        raise Exception("Invalid json directory location [%s]" % args.json_dir)

    if not os.path.isdir(args.pdb_dir):
        raise Exception("Invalid pdb directory location [%s]" % args.pdb_dir)

    if not os.path.isdir(os.path.dirname(args.output)):
        raise Exception("Output directory does not exists [%s]" % os.path.dirname(args.output))

    if os.path.isfile(args.output):
        warnings.warn("output file will be overwritten [%s]" % args.output)

    amino = sd.get_amino(args.amino.upper())

    def check_validity(f):
        return f.endswith('.json') and \
               os.path.isfile(os.path.join(args.pdb_dir, '%s.pdb' % (f[:-5])))

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(os.path.basename(os.path.abspath(__file__)))
    json_files = [os.path.join(args.json_dir, f) for f in os.listdir(args.json_dir) if check_validity(f)]
    pdb_files = [os.path.join(args.pdb_dir, '%s.pdb' % os.path.basename(f)[:-5]) for f in json_files]
    assert len(json_files) == len(pdb_files)
    logger.info("Number of files to process: %d" % len(json_files))

    sasa_info = cg.filtered_sasa(json_files, str(amino), area_type=[args.area_type])
    logger.info("All sasa files are read!!")

    all_features = None
    for i, filename in enumerate(pdb_files):
        logger.info("Processing %d-th file [%s]" % (i+1, filename))
        if i not in sasa_info:
            continue
        result = sasa_info[i]
        assert len(result) == 1
        chain = list(result.keys())[0]
        resids = [int(r) for r in result[chain].keys()]
        ca_trace = sd.read_trajectory_catrace(filename)
        ca_trace = list(ca_trace[0].values())[0]
        n = len(ca_trace)
        if n < args.neighbors:
            continue
        all_res = ca_trace.residue_ids
        resids = [r for r in resids if (r in all_res) and (all_res.index(r) > 0) and (all_res.index(r) < (n - 1))]
        if len(args.properties) > 0:
            sig = cg.cg_neighbor_signature(ca_trace, resids, properties=args.properties)
        else:
            sig = cg.cg_neighbor_signature(ca_trace, resids)
        assert len(sig) == len(resids)
        for j, r in enumerate(resids):
            sig[j].append(result[chain][str(r)][args.area_type])
        sig = np.asanyarray(sig)
        if len(sig.shape) != 2:
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


