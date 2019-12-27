#

# clocks

create_clock -period "125 MHz" [ get_ports si42_clk_125 ]
create_clock -period "50 MHz" [ get_ports si42_clk_50 ]

# si5345
create_clock -period "125 MHz" [ get_ports clk_125_bottom ]
create_clock -period "125 MHz" [ get_ports clk_125_top ]

create_clock -period "50 MHz" [ get_ports clk_aux ]
create_clock -period "156.25 MHz" [ get_ports qsfp_pll_clk ]
create_clock -period "125 MHz" [ get_ports pod_pll_clk ]



derive_pll_clocks -create_base_clocks
derive_clock_uncertainty

#run counter signal to data output - where to put this?
set_false_path -from {fe_block:e_fe_block|resetsys:e_reset_system|state_controller:i_state_controller|runnumber[*]} -to {fe_block:e_fe_block|data_merger:e_merger|data_out[*]}
set_false_path -from {fe_block:e_fe_block|resetsys:e_reset_system|state_controller:i_state_controller|runnumber[*]} -to {fe_block:e_fe_block|data_merger:e_merger_secondary|data_out[*]}

#set_false_path -to [ get_ports {LED[*]} ]
