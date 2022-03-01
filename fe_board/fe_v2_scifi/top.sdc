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

# false paths
set_false_path -from {scifi_path:e_tile_path|mutrig_datapath:e_mutrig_datapath|receiver_block:u_rxdeser|data_decoder:\gen_channels:*:datadec|ready_buf} -to {*}
set_false_path -from {*} -to {scifi_path:e_tile_path|doNotCompileAwayMux:dnca|o_led}
set_false_path -from {*} -to {scifi_path:e_tile_path|miso_156}