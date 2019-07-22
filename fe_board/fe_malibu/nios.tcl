package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 2

source {../fe/nios_avm.tcl}

add_avalon_proxy avm_test 0x70040000 16 1

add_connection avm_clk.clk       avm_test.clk
add_connection avm_clk.clk_reset avm_test.reset

source nios_mscb_inc.tcl

save_system {nios.qsys}
