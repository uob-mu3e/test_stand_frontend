# FEB_common false paths
# M. Mueller, September 2020

# idk why this is required, sync chain is not recognised
set_false_path -from {fe_block_v2:e_fe_block|data_merger:e_merger|terminated[0]} -to {fe_block_v2:e_fe_block|resetsys:e_reset_system|ff_sync:i_ff_sync|ff[0][0]}

# buttons
set_false_path -from {debouncer:db1|o_q[0]} -to {*}
set_false_path -from {debouncer:db2|o_q[0]} -to {*}

# Arria internal temperature
set_false_path -from {fe_block_v2:e_fe_block|nios:e_nios|nios_temp:temp*} -to {fe_block_v2:e_fe_block|fe_reg.rdata*}
set_false_path -from {fe_block_v2:e_fe_block|arriaV_temperature_ce} -to {*}

# this one is tricky, it's not really a false path but i think we also cannot sync to clk_reco (we can, but might screw up reset alignment)
set_false_path -from {fe_block_v2:e_fe_block|firefly:firefly|lvds_controller:e_lvds_controller|o_dpa_lock_reset} -to {fe_block_v2:e_fe_block|firefly:firefly|lvds_rx:lvds_rx_inst0*}