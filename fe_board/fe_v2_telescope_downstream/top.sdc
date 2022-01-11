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
create_clock -period "625.000 MHz" [ get_ports clk_125_bottom ]
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

# SPI Input/Output delays max10 spi
set_input_delay -clock { spare_clk_osc } -min 2 [get_ports {max10_spi_mosi}]
set_input_delay -clock { spare_clk_osc } -min 2 [get_ports {max10_spi_miso}]
set_input_delay -clock { spare_clk_osc } -min 2 [get_ports {max10_spi_D1}]
set_input_delay -clock { spare_clk_osc } -min 2 [get_ports {max10_spi_D2}]

set_input_delay -clock { spare_clk_osc } -max 3 [get_ports {max10_spi_mosi}]
set_input_delay -clock { spare_clk_osc } -max 3 [get_ports {max10_spi_miso}]
set_input_delay -clock { spare_clk_osc } -max 3 [get_ports {max10_spi_D1}]
set_input_delay -clock { spare_clk_osc } -max 3 [get_ports {max10_spi_D2}]

set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_sclk}]
set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_mosi}]
set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_miso}]
set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_D1}]
set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_D2}]
set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_D3}]
set_output_delay -clock { spare_clk_osc } -min 0.5 [get_ports {max10_spi_csn}]

set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_sclk}]
set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_mosi}]
set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_miso}]
set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_D1}]
set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_D2}]
set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_D3}]
set_output_delay -clock { spare_clk_osc } -max 0 [get_ports {max10_spi_csn}]

set_false_path -from {testcounter*} -to {fe_block_v2:e_fe_block|feb_reg_mapping:e_reg_mapping|testout*}
set_false_path -from {run_state_125_reg*} -to {run_state_625_reg*}
set_false_path -from {trig1_buffer_125} -to {trig1_buffer_125_reg}
set_false_path -from {trig0_buffer_125} -to {trig0_buffer_125_reg}
set_false_path -from {trig1_timestamp_save*} -to {trig1_ts_final*}
set_false_path -from {trig0_timestamp_save*} -to {trig0_ts_final*}
set_false_path -from {trig0_buffer_125_b} -to {trig0_buffer_125_reg_b}
set_false_path -from {trig1_buffer_125_b} -to {trig1_buffer_125_reg_b}
