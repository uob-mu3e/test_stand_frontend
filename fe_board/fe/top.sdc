#

# clocks

create_clock -period "125 MHz" [ get_ports si42_clk_125 ]
create_clock -period "50 MHz" [ get_ports si42_clk_50 ]

create_clock -period "125 MHz" [ get_ports clk_aux ]
create_clock -period "156.25 MHz" [ get_ports qsfp_pll_clk ]
create_clock -period "125 MHz" [ get_ports pod_pll_clk ]



derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



#set_false_path -to [ get_ports {LED[*]} ]
