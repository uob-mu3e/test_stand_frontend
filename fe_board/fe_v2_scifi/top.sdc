# global clock constrains for feb_v2
# M. Mueller, September 2020

# Trick : Setting the clocks to somewhat strange frequencies -> timing analyzer warnings for domain transitions
create_clock -period "125.004 MHz" [ get_ports LVDS_clk_si1_fpga_A ]
create_clock -period "125.003 MHz" [ get_ports LVDS_clk_si1_fpga_B ]
create_clock -period "156.253 MHz" [ get_ports transceiver_pll_clock[0] ]
create_clock -period "156.252 MHz" [ get_ports transceiver_pll_clock[1] ]
create_clock -period "156.251 MHz" [ get_ports transceiver_pll_clock[2] ]
create_clock -period "125.003 MHz" [ get_ports lvds_firefly_clk ]
create_clock -period  "50.003 MHz" [ get_ports systemclock ]
create_clock -period  "50.002 MHz" [ get_ports systemclock_bottom ]
create_clock -period "125.001 MHz" [ get_ports clk_125_top ]
create_clock -period "125.002 MHz" [ get_ports clk_125_bottom ]
create_clock -period  "50.001 MHz" [ get_ports spare_clk_osc ]


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
set_false_path -from {*} -to {scifi_path:e_scifi_path|miso_156}
set_false_path -from {scifi_path:e_scifi_path|scifi_reg_mapping:e_scifi_reg_mapping|ff_sync:e_cntreg_ctrl|ff[*][*]} -to {scifi_path:e_scifi_path|clkdiv_dynamic:e_test_pulse|cnt2_odd[*]}
set_false_path -from {scifi_path:e_scifi_path|scifi_reg_mapping:e_scifi_reg_mapping|ff_sync:e_cntreg_ctrl|ff[*][*]} -to {scifi_path:e_scifi_path|clkdiv_dynamic:e_test_pulse|clk2_odd}
