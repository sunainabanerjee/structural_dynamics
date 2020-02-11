import os
from structural_dynamics import *

if __name__ == "__main__":
    pdb_file = os.path.join( os.path.dirname(__file__),
                             'data',
                             'structure_1a1v.pdb')
    atp_site_residues = [202, 203, 204, 205, 206, 207, 208, 209,
                         210, 211, 212, 213, 214, 215, 227, 228,
                         229, 230, 231, 232, 233, 234, 235, 236,
                         237, 238, 239, 241, 242, 244, 245, 246,
                         267, 270, 288, 289, 290, 291, 292, 293,
                         294, 296, 299, 320, 321, 322, 323, 324,
                         325, 328, 329, 330, 331, 332, 333, 334,
                         335, 336, 365, 367, 411, 412, 413, 414,
                         415, 416, 417, 418, 419, 420, 421, 422,
                         425, 455, 456, 457, 458, 459, 460, 461,
                         462, 463, 464, 465, 466, 467, 468, 469,
                         473]
    rna_site_residues = [229, 230, 231, 232, 233, 234, 235, 253,
                         254, 255, 256, 268, 269, 270, 271, 272,
                         273, 274, 275, 291, 294, 295, 296, 297,
                         298, 299, 300, 301, 302, 303, 391, 392,
                         393, 394, 411, 412, 413, 414, 415, 432,
                         433, 434, 446, 448, 489, 490, 491, 492,
                         493, 494, 495, 496, 497, 498, 499, 500,
                         501, 502, 503, 504, 536, 547, 548, 549,
                         550, 551, 552, 553, 554, 555, 556, 557,
                         558, 559, 560, 561, 580, 581, 602]
    pdb = read_pdb(pdb_file=pdb_file)[0]['A']
    aligned_trace, min_xyz, max_xyz = align_to_xyz(pdb, residue_list=atp_site_residues)
    out_aligned_file = os.path.join(os.path.dirname(__file__),
                                    'out', '1a1v_atpn_aligned.pdb')
    out_boundingbox_file = os.path.join(os.path.dirname(__file__),
                                        'out', '1a1v_atpn_boundingbox.csv')

    with open(out_aligned_file, 'w+') as fp:
        aligned_trace.write(fp)

    with open(out_boundingbox_file, 'w+') as fp:
        fp.write("%s,%.3f,%.3f,%.3f\n" % ("min", min_xyz.x, min_xyz.y, min_xyz.z))
        fp.write("%s,%.3f,%.3f,%.3f\n" % ("max", max_xyz.x, max_xyz.y, max_xyz.z))

