#

add_instance                 avm_clk clock_source
set_instance_parameter_value avm_clk {clockFrequency} {156250000}
set_instance_parameter_value avm_clk {resetSynchronousEdges} {DEASSERT}

add_interface          avm_clk clock sink
set_interface_property avm_clk EXPORT_OF avm_clk.clk_in
add_interface          avm_reset reset sink
set_interface_property avm_reset EXPORT_OF avm_clk.clk_in_reset

if 1 {
    set name avm_qsfp
    add_instance ${name} avalon_proxy
    set_instance_parameter_value ${name} {addr_width} {16}

    add_connection clk.clk ${name}.clk
    add_connection clk.clk_reset ${name}.reset
    add_connection cpu.data_master ${name}.slave
    set_connection_parameter_value cpu.data_master/${name}.slave baseAddress {0x70010000}

    add_interface ${name} avalon master
    set_interface_property ${name} EXPORT_OF ${name}.master
}

if 1 {
    set name avm_pod
    add_instance ${name} avalon_proxy
    set_instance_parameter_value ${name} {addr_width} {16}

    add_connection clk.clk ${name}.clk
    add_connection clk.clk_reset ${name}.reset
    add_connection cpu.data_master ${name}.slave
    set_connection_parameter_value cpu.data_master/${name}.slave baseAddress {0x70020000}

    add_interface ${name} avalon master
    set_interface_property ${name} EXPORT_OF ${name}.master
}

if 1 {
    set name avm_sc
    add_instance ${name} avalon_proxy
    set_instance_parameter_value ${name} {addr_width} {18}
    set_instance_parameter_value ${name} {readLatency} {1}

    add_connection clk.clk ${name}.clk
    add_connection clk.clk_reset ${name}.reset
    add_connection cpu.data_master ${name}.slave
    set_connection_parameter_value cpu.data_master/${name}.slave baseAddress {0x70080000}

    add_interface ${name} avalon master
    set_interface_property ${name} EXPORT_OF ${name}.master
}

add_connection avm_clk.clk       avm_sc.clk
add_connection avm_clk.clk_reset avm_sc.reset
