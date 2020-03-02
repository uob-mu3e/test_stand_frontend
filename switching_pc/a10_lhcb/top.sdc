#

# internal oscillator
create_clock -period "100.0 MHz" [get_ports CLK_A10_100MHZ_P]

# PODs
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_0]

#
create_clock -period "40.0 MHz" [get_ports A10_SI53340_2_CLK_40_P]


derive_pll_clocks -create_base_clocks

derive_clock_uncertainty
