#

# set rx_clk of the reset to a 50% phase shift relative to clk_aux   (rising 4.000, falling 8.000)
create_clock -name {pod_pll_clk} -period 8.000 -waveform {4.000 8.000} [get_ports pod_pll_clk ]
