package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 2

source {../fe/nios_avm.tcl}

if 1 {
    set name avm_test
    add_instance ${name} avalon_proxy
    set_instance_parameter_value ${name} {addr_width} {16}
    set_instance_parameter_value ${name} {readLatency} {1}

    add_connection clk.clk ${name}.clk
    add_connection clk.clk_reset ${name}.reset
    add_connection cpu.data_master ${name}.slave
    set_connection_parameter_value cpu.data_master/${name}.slave baseAddress {0x70040000}

    add_interface ${name} avalon master
    set_interface_property ${name} EXPORT_OF ${name}.master
}

add_connection avm_clk.clk       avm_test.clk
add_connection avm_clk.clk_reset avm_test.reset

source nios_mscb_inc.tcl
source nios_reset_bypass_inc.tcl

save_system {nios.qsys}
