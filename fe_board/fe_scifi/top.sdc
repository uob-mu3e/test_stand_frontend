#false path for slow control registers (156MHz) to receiver clock (125M)
#signals are quasi-static and have synchronizers
set_false_path -through {scifi_path:e_scifi_path|s_subdet_reset_reg[*]}
set_false_path -through {scifi_path:e_scifi_path|s_dummyctrl_reg[*]}
set_false_path -through {scifi_path:e_scifi_path|s_dpctrl_reg[*]}

#clock domain crossing 125MHz lvds-pll clock  --> core clock
set_false_path -to {scifi_path:e_scifi_path|mutrig_datapath:e_mutrig_datapath|mutrig_store:\rcv_fifo:*:u_elastic_buffer|s_fifoused_reg[*]}


#reset alignment block: fals paths between slow and fast shifted clock, user needs to find proper dynamic phase shift and cycle settings
set_false_path -from [get_clocks {clk_125_top}] -to [get_clocks {e_scifi_path|u_resetshift|pll|altpll_component|auto_generated|pll1|clk[1]}]
set_false_path -from [get_clocks {clk_125_top}] -to [get_clocks {e_scifi_path|u_resetshift|pll|altpll_component|auto_generated|pll1|clk[0]}]
#todo: clock should be removed - we only want to have one clock
set_false_path -from [get_clocks {clk_aux}] -to [get_clocks {e_scifi_path|u_resetshift|pll|altpll_component|auto_generated|pll1|clk[0]}]
set_false_path -from [get_clocks {clk_aux}] -to [get_clocks {e_scifi_path|u_resetshift|pll|altpll_component|auto_generated|pll1|clk[1]}]

