#

# reset_sync
foreach e [ get_entity_instances reset_sync ] {
    set pins [ get_pins -compatibility_mode $e|*|clrn ]
#    set_max_delay -to $pins 100
#    set_min_delay -to $pins -100
#    set_max_skew -to $pins -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.8
    set_false_path -to $pins
}

# clk_phase
# see altera_std_synchronizer
foreach e [ get_entity_instances -nowarn "clk_phase" ] {
    set from_regs [ get_registers -nocase -nowarn "$e|d*" ]
    set to_regs [ get_registers -nocase -nowarn "$e|e_ff_sync|ff[0]*" ]
    if { [ get_collection_size $from_regs ] > 0 && [ get_collection_size $to_regs ] > 0 } {
        set_max_skew -to $from_regs -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.1
        # avoid set_false_path by setting min/max delays and max skew
        set_min_delay -from $from_regs -to $to_regs -100
        set_max_delay -from $from_regs -to $to_regs 100
        set_max_skew -from $from_regs -to $to_regs -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.1
#        set_false_path -from $from_regs -to $to_regs
    }
}
