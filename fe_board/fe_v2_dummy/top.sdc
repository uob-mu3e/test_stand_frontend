#

create_clock -period "125 MHz" [ get_ports LVDS_clk_si1_fpga_A ]
create_clock -period "125 MHz" [ get_ports LVDS_clk_si1_fpga_B ]
create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[0] ]
create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[1] ]
create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[2] ]
create_clock -period "125 MHz" [ get_ports lvds_firefly_clk ]
create_clock -period "50 MHz" [ get_ports systemclock ]
create_clock -period "50 MHz" [ get_ports systemclock_bottom ]
create_clock -period "125 MHz" [ get_ports clk_125_top ]
create_clock -period "125 MHz" [ get_ports clk_125_bottom ]
create_clock -period "50 MHz" [ get_ports spare_clk_osc ]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty

# does not detect i_ff_sync as sync chain .. why ?
set_false_path -from {fe_block_v2:e_fe_block|data_merger:e_merger|terminated[0]} -to {fe_block_v2:e_fe_block|resetsys:e_reset_system|ff_sync:i_ff_sync|ff[0][0]} 
