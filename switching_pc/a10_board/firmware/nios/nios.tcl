package require qsys

create_system {nios}
source {device.tcl}

source {../util/nios_base.tcl}

# ram 
set_instance_parameter_value ram {memorySize} {0x00080000}

# flash
add_instance flash a10_flash1616
add_connection clk.clk flash.clk
add_connection clk.clk_reset flash.reset
add_connection cpu.data_master flash.uas
set_connection_parameter_value cpu.data_master/flash.uas baseAddress {0x00000000}
add_connection cpu.instruction_master flash.uas
set_connection_parameter_value cpu.instruction_master/flash.uas baseAddress {0x00000000}
add_connection jtag_master.master flash.uas
add_connection cpu.debug_reset_request flash.reset
set_interface_property flash EXPORT_OF flash.out

# rx data

add_instance clk_rx clock_source
set_instance_parameter_value clk_rx {clockFrequency} 15625000
set_instance_parameter_value clk_rx {resetSynchronousEdges} {DEASSERT}

add_instance rx altera_avalon_pio
set_instance_parameter_value rx {direction} {Input}
set_instance_parameter_value rx {width} {32}

add_connection clk_rx.clk       rx.clk
add_connection clk_rx.clk_reset rx.reset

add_connection                 cpu.data_master rx.s1
set_connection_parameter_value cpu.data_master/rx.s1 baseAddress {0x700F0400}

add_interface clk_rx clock sink
set_interface_property clk_rx EXPORT_OF clk_rx.clk_in
add_interface rst_rx reset sink
set_interface_property rst_rx EXPORT_OF clk_rx.clk_in_reset

add_interface rx conduit end
set_interface_property rx EXPORT_OF rx.external_connection


# tx data

add_instance clk_tx clock_source
set_instance_parameter_value clk_tx {clockFrequency} 15625000
set_instance_parameter_value clk_tx {resetSynchronousEdges} {DEASSERT}

add_instance tx altera_avalon_pio
set_instance_parameter_value tx {direction} {Input}
set_instance_parameter_value tx {width} {32}

add_connection clk_tx.clk       tx.clk
add_connection clk_tx.clk_reset tx.reset

add_connection                 cpu.data_master tx.s1
set_connection_parameter_value cpu.data_master/tx.s1 baseAddress {0x700F0420}

add_interface clk_tx clock sink
set_interface_property clk_tx EXPORT_OF clk_tx.clk_in
add_interface rst_tx reset sink
set_interface_property rst_tx EXPORT_OF clk_tx.clk_in_reset

add_interface tx conduit end
set_interface_property tx EXPORT_OF tx.external_connection


# debug

add_instance clk_debug clock_source
set_instance_parameter_value clk_debug {clockFrequency} 15625000
set_instance_parameter_value clk_debug {resetSynchronousEdges} {DEASSERT}

add_instance debug altera_avalon_pio
set_instance_parameter_value debug {direction} {Input}
set_instance_parameter_value debug {width} {32}

add_connection clk_debug.clk       debug.clk
add_connection clk_debug.clk_reset debug.reset

add_connection                 cpu.data_master debug.s1
set_connection_parameter_value cpu.data_master/debug.s1 baseAddress {0x700F0440}

add_interface clk_debug clock sink
set_interface_property clk_debug EXPORT_OF clk_debug.clk_in
add_interface rst_debug reset sink
set_interface_property rst_debug EXPORT_OF clk_debug.clk_in_reset

add_interface debug conduit end
set_interface_property debug EXPORT_OF debug.external_connection




save_system {nios.qsys}