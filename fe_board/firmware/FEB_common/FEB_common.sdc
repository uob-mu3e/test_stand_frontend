# FEB_common false paths
# M. Mueller, September 2020

# idk why this is required, sync chain is not recognised
set_false_path -from {fe_block_v2:e_fe_block|data_merger:e_merger|terminated[0]} -to {fe_block_v2:e_fe_block|resetsys:e_reset_system|ff_sync:i_ff_sync|ff[0][0]}

# buttons
set_false_path -from {debouncer:db1|o_q[0]} -to {*}
set_false_path -from {debouncer:db2|o_q[0]} -to {*}

# Arria internal temperature
set_false_path -from {fe_block_v2:e_fe_block|nios:e_nios|nios_temp:temp*} -to {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|arriaV_temperature*}
set_false_path -from {fe_block_v2:e_fe_block|arriaV_temperature*} -to {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|arriaV_temperature*}
set_false_path -from {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|o_arriaV_temperature_ce} -to {*}
set_false_path -from {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|o_arriaV_temperature_clr} -to {*}

# Max10 adc
set_false_path -from {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|adc_reg*} -to {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|adc_reg*}

# this one is tricky, it's not really a false path but i think we also cannot sync to clk_reco (we can, but might screw up reset alignment)
set_false_path -from {fe_block_v2:e_fe_block|firefly:firefly|lvds_controller:e_lvds_controller|o_dpa_lock_reset} -to {fe_block_v2:e_fe_block|firefly:firefly|lvds_rx:lvds_rx_inst0*}

# single bits only (program req and fifo aclr)
set_false_path -from {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|o_programming_ctrl*} -to {*}

# other stuff
set_false_path -from {fe_block_v2:e_fe_block|firefly:firefly|lvds_controller:e_lvds_controller|o_ready} -to {*}
set_false_path -from {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|max10_status*} -to {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|o_max10_status*}
set_false_path -from {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|max10_version*} -to {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|o_max10_version*}
set_false_path -from {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|programming_status*} -to {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|o_programming_status*}
set_false_path -from {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|o_programming_addr_ena} -to {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|programming_addr_ena_reg}
set_false_path -from {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|o_programming_addr*} -to {fe_block_v2:e_fe_block|max10_interface:e_max10_interface|max_spi_data_to_max*}
set_false_path -from {*} -to {*max10_interface|o_programming_status*}

# int run emergeny REMOVE THIS AGAIN
# no need for the nios to be able to read somehting ever (mupix feb only, do not merge !!!)
#set_false_path -from {fe_block_v2:e_fe_block|nios:e_nios*} -to {*_reg_mapping|o_reg_rdata*}
