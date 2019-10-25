#

# reset_sync
foreach e [ get_entity_instances reset_sync ] {
    set pins [ get_pins -compatibility_mode $e|*|clrn ]
#    set_max_delay -to $pins 100
#    set_min_delay -to $pins -100
#    set_max_skew -to $pins -get_skew_value_from_clock_period dst_clock_period -skew_value_multiplier 0.8
    set_false_path -to $pins
}
