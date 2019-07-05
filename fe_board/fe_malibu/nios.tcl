package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 2



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

add_instance                 sc_clk clock_source
set_instance_parameter_value sc_clk {clockFrequency} {156250000}
set_instance_parameter_value sc_clk {resetSynchronousEdges} {DEASSERT}

add_connection sc_clk.clk       avm_sc.clk
add_connection sc_clk.clk_reset avm_sc.reset

add_interface          sc_clk clock sink
set_interface_property sc_clk EXPORT_OF sc_clk.clk_in
add_interface          sc_reset reset sink
set_interface_property sc_reset EXPORT_OF sc_clk.clk_in_reset

source nios_mscb_inc.tcl



save_system {nios.qsys}
