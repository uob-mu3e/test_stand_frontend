#

package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00080000}
source {a10/a10_flash1616.tcl}

nios_base.add_clock_source clk_156 156250000 -clock_export clk_156_clock -reset_export clk_156_reset
nios_base.export_avm avm_qsfpA 14 0x70010000 -clk clk_156
nios_base.export_avm avm_qsfpB 14 0x70020000 -clk clk_156
nios_base.export_avm avm_qsfpC 14 0x70030000 -clk clk_156
nios_base.export_avm avm_qsfpD 14 0x70040000 -clk clk_156

save_system {nios.qsys}
