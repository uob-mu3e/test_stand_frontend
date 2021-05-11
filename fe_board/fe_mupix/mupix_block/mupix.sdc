# MuPix FEB false paths
# M. Mueller, November 2020

# these should be fine
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_lvds_invert} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_datagen_control*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|data_unpacker*:unpacker_single|o_hit_ena_counter*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hit_ena_counters_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_in_ena_counters*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_in_ena_counters_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_out_ena_cnt*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_out_ena_cnt_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|last_sorter_hit*} -to {*}

# hitsorter false paths (to be checked)
set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|countermemory*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|nintime*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_reg_rdata*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|noutoftime*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_reg_rdata*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|noverflow*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_reg_rdata*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|credits*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_reg_rdata*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|nout*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_reg_rdata*}

# TODO: get rid of this one ... was working on mp10sc and did not have time to close properly M.Mueller
set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125[7]} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|data_unpacker*}
set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125[6]} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|data_unpacker*}
