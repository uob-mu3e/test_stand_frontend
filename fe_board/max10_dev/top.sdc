#

create_clock -name CLOCK -period 20.000 [get_ports {CLOCK}]

derive_clock_uncertainty

derive_pll_clocks
