add_instance parallel_mscb_in altera_avalon_pio
set_instance_parameter_value parallel_mscb_in {direction} {Input}
set_instance_parameter_value parallel_mscb_in {width} {12}
set_instance_parameter_value parallel_mscb_in {bitModifyingOutReg} {0}

add_instance parallel_mscb_out altera_avalon_pio
set_instance_parameter_value parallel_mscb_out {direction} {Output}
set_instance_parameter_value parallel_mscb_out {width} {12}
set_instance_parameter_value parallel_mscb_out {bitModifyingOutReg} {1}

add_instance counter_in altera_avalon_pio
set_instance_parameter_value counter_in {direction} {Input}
set_instance_parameter_value counter_in {width} {16}
set_instance_parameter_value counter_in {bitModifyingOutReg} {0}

add_interface parallel_mscb_in conduit end
set_interface_property parallel_mscb_in EXPORT_OF parallel_mscb_in.external_connection

add_interface parallel_mscb_out conduit end
set_interface_property parallel_mscb_out EXPORT_OF parallel_mscb_out.external_connection

add_interface counter_in conduit end
set_interface_property counter_in EXPORT_OF counter_in.external_connection

foreach { name clk reset avalon addr } {
    parallel_mscb_in  clk reset s1 0x0300
    parallel_mscb_out clk reset s1 0x0320
    counter_in        clk reset s1 0x0340
} {
    add_connection clk.clk       $name.$clk
    add_connection clk.clk_reset $name.$reset
    add_connection                 cpu.data_master $name.$avalon
    set_connection_parameter_value cpu.data_master/$name.$avalon baseAddress [ expr 0x700F0000 + $addr ]
    add_connection cpu.debug_reset_request $name.$reset
}
