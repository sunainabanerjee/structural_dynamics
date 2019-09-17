import os
from coarse_graining import *
from structural_dynamics import *

__version__ = "1.0"

if __name__ == "__main__":
    pdb_file = os.path.join(os.path.dirname(__file__), 'data', 'structure.pdb')
    assert os.path.isfile(pdb_file)
    pdb = read_pdb(pdb_file)[0]['B']
    ca_trace = pdb_to_catrace(pdb)
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'sasa_models')
    assert os.path.isdir(model_dir)
    predictor = SASAPredictor(model_folder=model_dir, model_type='xgb')
    resid, sasa = predictor.predict_sasa(ca_trace, ca_trace.residue_ids[2:-2])
    print(sasa)
