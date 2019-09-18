#

foreach e [ get_entity_instances xcvr_s4 ] {
    set regs [ get_registers ${e}|av_ctrl.readdata* ]
    set fanins [ get_fanins $regs ]

    set_max_delay -from $fanins -to $regs 100
    set_min_delay -from $fanins -to $regs -100
    set_net_delay -from $fanins -to $regs -max -get_value_from_clock_period dst_clock_period -value_multiplier 0.8
    set_max_skew -to $regs -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.8
}
