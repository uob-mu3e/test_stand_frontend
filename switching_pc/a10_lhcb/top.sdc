#

create_clock -period  "100.0 MHz" [get_ports CLK_A10_100MHZ_P]

derive_pll_clocks -create_base_clocks

derive_clock_uncertainty
