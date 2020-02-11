#! /bin/bash

export PYTHONPATH=/home/sumanta/PycharmProjects/structural_dynamics

# python3.6 generate_site_signature_library_post_augmentation.py --ref-pdb site_signature/1a1v_rna_site_aligned.pdb --box site_signature/1a1v_atp_site_boundingbox.csv --top-dir /home/sumanta/Project/structural_dynamics/coarsegrained/martini/scripts/augmentation/nma --out site_signature/atp_feature_vector_jan2020.csv

python3.6 generate_site_signature_library_post_augmentation.py --ref-pdb site_signature/1a1v_rna_site_aligned.pdb --box site_signature/1a1v_rna_site_boundingbox.csv --top-dir /home/sumanta/Project/structural_dynamics/coarsegrained/martini/scripts/augmentation/nma --out site_signature/rna_feature_vector_jan2020.csv
