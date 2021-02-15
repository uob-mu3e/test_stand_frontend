#

source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00080000}

# add dummy flash_tcm_* ports
add_instance flash a10_flash1616
add_interface flash conduit end
set_interface_property flash EXPORT_OF flash.out



nios_base.add_clock_source clk_pod 125 -clock_export avm_pod_clock -reset_export avm_pod_reset
nios_base.export_avm avm_pod 17 0x70100000 -clk clk_pod



nios_base.add_pio i2c_cs 32 Output 0x700F0260
