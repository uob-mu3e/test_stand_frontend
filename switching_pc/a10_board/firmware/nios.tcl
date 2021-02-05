#

source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00080000}
source {util/a10_flash1616.tcl}
set_instance_parameter_value spi numberOfSlaves 32

nios_base.add_clock_source avm_clk 156250000 -clock_export avm_clock -reset_export avm_reset
nios_base.export_avm avm_xcvr 16 0x70040000 -clk avm_clk
