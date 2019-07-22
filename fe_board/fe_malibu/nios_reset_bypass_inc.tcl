#

add_instance reset_bypass_out altera_avalon_pio
set_instance_parameter_value reset_bypass_out {direction} {Output}
set_instance_parameter_value reset_bypass_out {width} {12}
set_instance_parameter_value reset_bypass_out {bitModifyingOutReg} {1}

add_interface reset_bypass_out conduit end
set_interface_property reset_bypass_out EXPORT_OF reset_bypass_out.external_connection

foreach { name clk reset avalon addr } {
    reset_bypass_out clk reset s1 0x0360
} {
    add_connection clk.clk       $name.$clk
    add_connection clk.clk_reset $name.$reset
    add_connection                 cpu.data_master $name.$avalon
    set_connection_parameter_value cpu.data_master/$name.$avalon baseAddress [ expr 0x700F0000 + $addr ]
    add_connection cpu.debug_reset_request $name.$reset
}
