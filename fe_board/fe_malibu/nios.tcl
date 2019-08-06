#

package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 16

source {../fe/nios_avm.tcl}

nios_base.export_avm avm 14 0x70040000 -addressUnits 32 -readLatency 1

add_connection avm_clk.clk       avm.clk
add_connection avm_clk.clk_reset avm.reset

source ../firmware/FEB_common/nios_mscb_inc.tcl
source nios_reset_bypass_inc.tcl

save_system {nios.qsys}
