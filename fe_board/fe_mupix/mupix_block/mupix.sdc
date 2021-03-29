# MuPix FEB false paths
# M. Mueller, November 2020

# some control regs should be mostly constant
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|o_mp_datagen_control*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|o_mp_readout_mode*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|o_mp_lvds_link_mask*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_reg_mapping:e_mupix_reg_mapping|o_mp_readout_mode*} -to {*}

# TODO: get rid of this one ... was working on mp10sc and did not have time to close properly M.Mueller
set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125[7]} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|data_unpacker*}
set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125[6]} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|data_unpacker*}