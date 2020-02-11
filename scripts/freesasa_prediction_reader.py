import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert freesasa json output to tsv")

    parser.add_argument('--json', action='store', dest='json_file',
                        type=str, required=True,
                        help="json file against which the prediction will be performed")

    args = parser.parse_args()

    assert os.path.isfile(args.json_file)

    with open(args.json_file, 'r') as fp:
        data = json.load(fp)

    assert isinstance(data, dict)
    assert 'results' in data
    assert len(data['results']) > 0
    for result in data['results']:
        assert isinstance(result, dict)
        assert 'structure' in result
        for structure in result['structure']:
            assert isinstance(structure, dict)
            assert 'chains' in structure
            for chain in structure['chains']:
                assert 'residues' in chain
                for residue in chain['residues']:
                    print("%d\t%s\t%.2f" % (int(residue['number']),
                                             residue['name'],
                                             float(residue['area']['total'])))

