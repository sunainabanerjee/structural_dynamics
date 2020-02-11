load 1a1v_atp_centered.pdb

hide all
util.ss
show cartoon

select seg1, resi 230-240
cmd.show("spheres","seg1")
util.color_deep("red", 'seg1', 0)

