#

package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00080000}



nios_base.add_clock_source clk_pod 100 -clock_export avm_pod_clock -reset_export avm_pod_reset
nios_base.export_avm avm_pod 14 0x70010000 -clk clk_pod



save_system {nios.qsys}
