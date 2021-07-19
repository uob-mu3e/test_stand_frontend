#

foreach e [ concat \
    [ get_entity_instances -nowarn xcvr_a10 ] \
    [ get_entity_instances -nowarn xcvr_enh ] \
    [ get_entity_instances -nowarn xcvr_sfp ] \
] {
    set regs [ get_registers ${e}|av_ctrl.readdata* ]
    set fanins [ get_fanins $regs ]

    set_min_delay -from $fanins -to $regs -100
    set_max_delay -from $fanins -to $regs 100
}
