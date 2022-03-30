#

# oscillator
create_clock -period "100.001 MHz" [get_ports CLK_A10_100MHZ_P]

# SI53344
create_clock -period "125.002 MHz" [get_ports A10_CUSTOM_CLK_P]

# SI5345_1
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_0]
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_1]
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_2]
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_3]

# SI5345_2
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_4]
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_5]
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_6]
create_clock -period "125.000 MHz" [get_ports A10_REFCLK_GBT_P_7]

derive_pll_clocks -create_base_clocks

derive_clock_uncertainty
