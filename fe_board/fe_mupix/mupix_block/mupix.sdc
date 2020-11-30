# MuPix FEB false paths
# M. Mueller, November 2020

# some control regs should be mostly constant
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|mp_datagen_control*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|mp_readout_mode*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|mp_lvds_link_mask*} -to {*}