import os
import math
import pandas as pd
from structural_dynamics import *

if __name__ == "__main__":
    pdb_dir = os.path.join(os.path.dirname(__file__),
                           "data",
                           "structures")
    assert os.path.isdir(pdb_dir)
    pdb_files = [ os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    data = {'amino': [],
            'atom': [],
            'distance': [],
            'angle': [],
            'dihedral': []}
    failed = []
    for pdb_file in pdb_files:
        print("Processing: %s" % pdb_file)
        for chain, pdb in read_pdb(pdb_file)[0].items():
            try:
                result = reconstruction_units(pdb)
                for r in result:
                    for aa in result[r]:
                        residue, dist, ang, dihed = result[r][aa]
                        if dihed < 0:
                            dihed = dihed + 2 * math.pi
                        data['amino'].append(residue)
                        data['atom'].append(aa)
                        data['distance'].append(dist)
                        data['angle'].append(ang)
                        data['dihedral'].append(dihed)
            except:
               failed.append(pdb_file)
    data = pd.DataFrame.from_dict(data)
    outfile = os.path.join(os.path.dirname(__file__), 'out', 'dihed_stat.csv')
    data.to_csv(outfile, float_format="%.3f", index=False, header=True)
