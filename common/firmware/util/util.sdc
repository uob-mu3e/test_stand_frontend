#

# ff_sync
# see altera handshake_clock_crosser
foreach e [ get_entity_instances -nowarn "ff_sync" ] {
    set to_regs [ get_registers -nocase -nowarn "$e|ff*" ]
    if { [ get_collection_size $to_regs ] > 1 } {
        set_false_path -to $to_regs
#        set_min_delay -to $to_regs -100
#        set_max_delay -to $to_regs 100
#        set_max_skew -to $to_regs -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.5
    }
}

# reset_sync
# see altera_reset_controller.sdc
foreach e [ get_entity_instances -nowarn "reset_sync" ] {
    set aclr_pins [ get_pins -compatibility_mode -nocase -nowarn "$e|*|aclr" ]
    if { [ get_collection_size $aclr_pins ] > 0 } {
        set_false_path -to $aclr_pins
    }
    set clrn_pins [ get_pins -compatibility_mode -nocase -nowarn "$e|*|clrn" ]
    if { [ get_collection_size $clrn_pins ] > 0 } {
        set_false_path -to $clrn_pins
#        set_min_delay -to $clrn_pins -100
#        set_max_delay -to $clrn_pins 100
#        set_max_skew -to $clrn_pins -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.8
    }
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
