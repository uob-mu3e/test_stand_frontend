# global clock constrains for feb_v2
# M. Mueller, September 2020

# Trick : Setting the clocks to somewhat strange frequencies -> timing analyzer warnings for domain transitions
create_clock -period "125.004 MHz" [ get_ports LVDS_clk_si1_fpga_A ]
create_clock -period "125.003 MHz" [ get_ports LVDS_clk_si1_fpga_B ]
create_clock -period "156.2503 MHz" [ get_ports transceiver_pll_clock[0] ]
create_clock -period "156.2502 MHz" [ get_ports transceiver_pll_clock[1] ]
create_clock -period "156.2501 MHz" [ get_ports transceiver_pll_clock[2] ]
create_clock -period "125.003 MHz" [ get_ports lvds_firefly_clk ]
create_clock -period "50.03 MHz" [ get_ports systemclock ]
create_clock -period "50.02 MHz" [ get_ports systemclock_bottom ]
create_clock -period "125.001 MHz" [ get_ports clk_125_top ]
create_clock -period "125.002 MHz" [ get_ports clk_125_bottom ]
create_clock -period "50.01 MHz" [ get_ports spare_clk_osc ]


# Do this when you are done with timing:

# create all the clocks coming from Si-Chip or oscillator
#create_clock -period "125 MHz" [ get_ports LVDS_clk_si1_fpga_A ]
#create_clock -period "125 MHz" [ get_ports LVDS_clk_si1_fpga_B ]
#create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[0] ]
#create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[1] ]
#create_clock -period "156.25 MHz" [ get_ports transceiver_pll_clock[2] ]
#create_clock -period "125 MHz" [ get_ports lvds_firefly_clk ]
#create_clock -period "50 MHz" [ get_ports systemclock ]
#create_clock -period "50 MHz" [ get_ports systemclock_bottom ]
#create_clock -period "125 MHz" [ get_ports clk_125_top ]
#create_clock -period "125 MHz" [ get_ports clk_125_bottom ]
#create_clock -period "50 MHz" [ get_ports spare_clk_osc ]

# derive pll clocks from base clocks
derive_pll_clocks -create_base_clocks
derive_clock_uncertainty

# false paths
set_false_path -from {fe_block_v2:e_fe_block|data_merger:e_merger|terminated[0]} -to {fe_block_v2:e_fe_block|resetsys:e_reset_system|ff_sync:i_ff_sync|ff[0][0]}

# this one is tricky, it's not really a false path but i think we also cannot sync to clk_reco (we can, but might screw up reset alignment)
set_false_path -from {fe_block_v2:e_fe_block|firefly:firefly|lvds_controller:e_lvds_controller|o_dpa_lock_reset} -to {fe_block_v2:e_fe_block|firefly:firefly|lvds_rx:lvds_rx_inst0*}