#

# clocks

create_clock -period "40 MHz" [ get_ports si42_clk_40 ]
create_clock -period "80 MHz" [ get_ports si42_clk_80 ]

#for the mutrig FEs the Si output is currently configured to 625MHz
create_clock -period "625 MHz" [ get_ports clk_125_top ]

create_clock -period "125 MHz" [ get_ports clk_125_bottom ]
create_clock -period "125 MHz" [ get_ports clk_aux ]
create_clock -period "156.25 MHz" [ get_ports qsfp_pll_clk ]
create_clock -period "125 MHz" [ get_ports pod_pll_clk ]



derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



#set_false_path -to [ get_ports {LED[*]} ]
