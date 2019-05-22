package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 2



add_instance clk_data clock_source
set_instance_parameter_value clk_data {clockFrequency} 16000000
set_instance_parameter_value clk_data {resetSynchronousEdges} {DEASSERT}

add_instance pio_in altera_avalon_pio
set_instance_parameter_value pio_in {direction} {Input}
set_instance_parameter_value pio_in {width} {32}

add_instance pio_out altera_avalon_pio
set_instance_parameter_value pio_out {direction} {Output}
set_instance_parameter_value pio_out {width} {32}

add_connection clk_data.clk       pio_in.clk
add_connection clk_data.clk_reset pio_in.reset

add_connection clk_data.clk       pio_out.clk
add_connection clk_data.clk_reset pio_out.reset

add_connection                 cpu.data_master pio_in.s1
set_connection_parameter_value cpu.data_master/pio_in.s1 baseAddress {0x700F0400}

add_connection                 cpu.data_master pio_out.s1
set_connection_parameter_value cpu.data_master/pio_out.s1 baseAddress {0x700F0420}

add_interface clk_data clock sink
set_interface_property clk_data EXPORT_OF clk_data.clk_in
add_interface rst_data reset sink
set_interface_property rst_data EXPORT_OF clk_data.clk_in_reset

add_interface pio_in conduit end
set_interface_property pio_in EXPORT_OF pio_in.external_connection
add_interface pio_out conduit end
set_interface_property pio_out EXPORT_OF pio_out.external_connection



if 1 {
    set name avm_qsfp
    add_instance ${name} avalon_proxy
    set_instance_parameter_value ${name} {addr_width} {14}

    add_connection clk.clk ${name}.clk
    add_connection clk.clk_reset ${name}.reset
    add_connection cpu.data_master ${name}.slave
    set_connection_parameter_value cpu.data_master/${name}.slave baseAddress {0x70010000}

    add_interface ${name} avalon master
    set_interface_property ${name} EXPORT_OF ${name}.master
}

if 1 {
    set name avm_sc
    add_instance ${name} avalon_proxy
    set_instance_parameter_value ${name} {addr_width} {14}

    add_connection clk.clk ${name}.clk
    add_connection clk.clk_reset ${name}.reset
    add_connection cpu.data_master ${name}.slave
    set_connection_parameter_value cpu.data_master/${name}.slave baseAddress {0x70020000}

    add_interface ${name} avalon master
    set_interface_property ${name} EXPORT_OF ${name}.master
}



save_system {nios.qsys}
