#

# clocks

create_clock -period "125 MHz" [ get_ports clk_aux ]
create_clock -period "156.25 MHz" [ get_ports qsfp_pll_clk ]
create_clock -period "125 MHz" [ get_ports pod_pll_clk ]
create_clock -period "625 MHz" [ get_ports clk_625 ]



derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



#set_false_path -to [ get_ports {LED[*]} ]



# xcvr|av_ctrl.readdata
if 1 {
    set regs [ get_registers {fe_block:*|xcvr_s4:*|av_ctrl.readdata*} ]
    set_max_delay -from [ get_registers * ] -to $regs 100
    set_min_delay -from [ get_registers * ] -to $regs -100
    set_net_delay -from [ get_registers * ] -to $regs -max -get_value_from_clock_period dst_clock_period -value_multiplier 0.8
    set_max_skew -to $regs -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.8
}

# reset_sync|arst_n
if 1 {
    set pins [ get_pins -compatibility_mode -nowarn {reset_sync:*|*|clrn} ]
    if { [ get_collection_size $pins ] > 0 } { set_false_path -to $pins }
    set pins [ get_pins -compatibility_mode -nowarn {*|reset_sync:*|*|clrn} ]
    if { [ get_collection_size $pins ] > 0 } { set_false_path -to $pins }
    set pins [ get_pins -compatibility_mode -nowarn {reset_sync:*|*|aclr} ]
    if { [ get_collection_size $pins ] > 0 } { set_false_path -to $pins }
    set pins [ get_pins -compatibility_mode -nowarn {*|reset_sync:*|*|aclr} ]
    if { [ get_collection_size $pins ] > 0 } { set_false_path -to $pins }
}
