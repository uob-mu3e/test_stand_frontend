#

foreach e [ concat \
    [ get_entity_instances xcvr_a10 ] \
    [ get_entity_instances xcvr_enh ] \
] {
    set regs [ get_registers ${e}|av_ctrl.readdata* ]
    set fanins [ get_fanins $regs ]

    set_min_delay -from $fanins -to $regs -100
    set_max_delay -from $fanins -to $regs 100
}
