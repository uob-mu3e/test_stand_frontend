#

source "device.tcl"

add_instance emif_0 altera_emif

set_instance_parameter_value emif_0 {MEM_DDR4_DQ_WIDTH} {64}

set_instance_parameter_value emif_0 {MEM_DDR4_FORMAT_ENUM} {MEM_FORMAT_SODIMM}

set_instance_parameter_value emif_0 {PROTOCOL_ENUM} {PROTOCOL_DDR4}

set_instance_property emif_0 AUTO_EXPORT {true}
