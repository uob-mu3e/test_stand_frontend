# MuPix FEB false paths
# M. Mueller, November 2020

set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_lvds_invert} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_reset_n_lvds} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_datagen_control*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|data_unpacker*:unpacker_single|o_hit_ena_counter*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hit_ena_counters_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_in_ena_counters*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_in_ena_counters_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_out_ena_cnt*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_out_ena_cnt_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|last_sorter_hit*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix|rx_ready*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix|o_rx_ready*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|lvds_link_mask*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|lvds_link_mask_reg*}
set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_in_ena_counters*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_readout_mode*} -to {*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_mp_lvds_link_mask*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|lvds_link_mask_reg*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|reg_delay} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_sorter_inject*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|mp_sorter_inject*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|mupix_datapath_reg_mapping:e_mupix_datapath_reg_mapping|o_sorter_inject*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|data_decoder:\gendec:*:datadec|ready_buf} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|o_rx_ready*}

#set_false_path -from {fe_block_v2:e_fe_block|resetsys:e_reset_system|state_phase_box:i_state_phase_box|o_state_125*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|countermemory*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|nintime*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|mp_sorter_reg_mapping:e_mp_sorter_reg_mapping|nintime*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|noutoftime*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|mp_sorter_reg_mapping:e_mp_sorter_reg_mapping|noutoftime*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|noverflow*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|mp_sorter_reg_mapping:e_mp_sorter_reg_mapping|noverflow*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|credits*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|mp_sorter_reg_mapping:e_mp_sorter_reg_mapping|credit*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|nout*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|mp_sorter_reg_mapping:e_mp_sorter_reg_mapping|nout*}

set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|mp_sorter_reg_mapping:e_mp_sorter_reg_mapping|o_sorter_delay*} -to {*}

set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|countermemory:\genmem:*:gencmem:*:cmem|altsyncram:altsyncram_component|altsyncram_teu3:auto_generated|ram_block1a0~PORT_B_WRITE_ENABLE_REG} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|dcountertemp*}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|countermemory:\genmem:*:gencmem:*:cmem|altsyncram:altsyncram_component|altsyncram_teu3:auto_generated|ram_block1a0~PORT_B_WRITE_ENABLE_REG} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|hitsorter_wide:sorter|tocmem_hitwriter[*][*][*]}

set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|disp_err_counter[*][*]} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|o_rx_status[*][*]}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|data_decoder:\gendec:*:datadec|ready_buf} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|o_rx_status[*][*]}
set_false_path -from {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|data_decoder:\gendec:*:datadec|state_out*} -to {mupix_block:e_mupix_block|mupix_datapath:e_mupix_datapath|receiver_block_mupix:lvds_block|o_rx_status[*][*]}