import os
import regressor as reg
import coarse_graining as cg
import structural_dynamics as sd


if __name__ == "__main__":
    restype, chain_length = "ARG", 10
    model_file = os.path.join(os.path.dirname(__file__), 'data', 'arg_chain')
    pdb_file = os.path.join(os.path.dirname(__file__), 'data', 'structure.pdb')
    assert os.path.isdir(model_file) and os.path.isfile(pdb_file)
    ca_trace = sd.read_trajectory_catrace(pdb_file)[0]['B']
    model = reg.ChainedMLP(chain_length=chain_length, chained_incr=3)
    model.load(model_file)
    results = cg.place_atoms(ca_trace, model, sd.get_amino(restype))
    resids = sorted([int(k) for k in results.keys()])
    ca_coords = [ca_trace.xyz(r) for r in resids]
    entity, atom_id = [], 1
    for i, r in enumerate(resids):
        for atom, coords in results[r].items():
            entity.append({'resname': restype,
                           'resid': r,
                           'atomname': atom,
                           'atomid': atom_id,
                           'x': coords['x'],
                           'y': coords['y'],
                           'z': coords['z']
                           })
            atom_id += 1
        entity.append({'resname': restype,
                       'resid': r,
                       'atomname': "CA",
                       'atomid': atom_id,
                       'x': ca_coords[i][0],
                       'y': ca_coords[i][1],
                       'z': ca_coords[i][2]})
        atom_id += 1
    pdb_fixed = sd.PDBStructure(name="dummy", entry=entity)
    out_file = os.path.join(os.path.dirname(__file__), 'out', 'fixed_arg.pdb')
    with open(out_file, 'w') as fh:
        pdb_fixed.write(fh)

