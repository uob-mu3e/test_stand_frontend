package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00080000}
source {a10/a10_flash1616.tcl}



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



save_system {nios.qsys}
