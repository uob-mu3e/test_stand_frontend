#

package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 2

source {../fe/nios_avm.tcl}

nios_base.export_avm avm_test 0x70040000 16 -readLatency 1

add_connection avm_clk.clk       avm_test.clk
add_connection avm_clk.clk_reset avm_test.reset

source ../firmware/FEB_common/nios_mscb_inc.tcl
source nios_reset_bypass_inc.tcl

save_system {nios.qsys}
