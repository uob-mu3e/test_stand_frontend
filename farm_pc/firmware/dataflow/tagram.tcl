# qsys scripting (.tcl) file for tagram
package require -exact qsys 16.0

create_system {tagram}

set_project_property DEVICE_FAMILY {Arria 10}
set_project_property DEVICE {10AX115N2F45I1SG}
set_project_property HIDE_FROM_IP_CATALOG {true}

# Instances and instance parameters
# (disabled instances are intentionally culled)
add_instance ram_1port_0 ram_1port 18.1
set_instance_parameter_value ram_1port_0 {GUI_ADDRESSSTALL_A} {0}
set_instance_parameter_value ram_1port_0 {GUI_AclrAddr} {0}
set_instance_parameter_value ram_1port_0 {GUI_AclrByte} {0}
set_instance_parameter_value ram_1port_0 {GUI_AclrData} {0}
set_instance_parameter_value ram_1port_0 {GUI_AclrOutput} {0}
set_instance_parameter_value ram_1port_0 {GUI_BYTE_ENABLE} {0}
set_instance_parameter_value ram_1port_0 {GUI_BYTE_SIZE} {8}
set_instance_parameter_value ram_1port_0 {GUI_BlankMemory} {0}
set_instance_parameter_value ram_1port_0 {GUI_CLOCK_ENABLE_INPUT_A} {0}
set_instance_parameter_value ram_1port_0 {GUI_CLOCK_ENABLE_OUTPUT_A} {0}
set_instance_parameter_value ram_1port_0 {GUI_Clken} {0}
set_instance_parameter_value ram_1port_0 {GUI_FILE_REFERENCE} {0}
set_instance_parameter_value ram_1port_0 {GUI_FORCE_TO_ZERO} {0}
set_instance_parameter_value ram_1port_0 {GUI_IMPLEMENT_IN_LES} {0}
set_instance_parameter_value ram_1port_0 {GUI_INIT_FILE_LAYOUT} {PORT_A}
set_instance_parameter_value ram_1port_0 {GUI_INIT_TO_SIM_X} {0}
set_instance_parameter_value ram_1port_0 {GUI_JTAG_ENABLED} {0}
set_instance_parameter_value ram_1port_0 {GUI_JTAG_ID} {NONE}
set_instance_parameter_value ram_1port_0 {GUI_MAXIMUM_DEPTH} {0}
set_instance_parameter_value ram_1port_0 {GUI_MIFfilename} {}
set_instance_parameter_value ram_1port_0 {GUI_NUMWORDS_A} {65536}
set_instance_parameter_value ram_1port_0 {GUI_RAM_BLOCK_TYPE} {Auto}
set_instance_parameter_value ram_1port_0 {GUI_READ_DURING_WRITE_MODE_PORT_A} {0}
set_instance_parameter_value ram_1port_0 {GUI_RegAddr} {1}
set_instance_parameter_value ram_1port_0 {GUI_RegData} {1}
set_instance_parameter_value ram_1port_0 {GUI_RegOutput} {1}
set_instance_parameter_value ram_1port_0 {GUI_SclrOutput} {0}
set_instance_parameter_value ram_1port_0 {GUI_SingleClock} {0}
set_instance_parameter_value ram_1port_0 {GUI_TBENCH} {0}
set_instance_parameter_value ram_1port_0 {GUI_WIDTH_A} {26}
set_instance_parameter_value ram_1port_0 {GUI_WRCONTROL_ACLR_A} {0}
set_instance_parameter_value ram_1port_0 {GUI_X_MASK} {0}
set_instance_parameter_value ram_1port_0 {GUI_rden} {0}

# exported interfaces
set_instance_property ram_1port_0 AUTO_EXPORT {true}

# interconnect requirements
set_interconnect_requirement {$system} {qsys_mm.clockCrossingAdapter} {HANDSHAKE}
set_interconnect_requirement {$system} {qsys_mm.enableEccProtection} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.insertDefaultSlave} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.maxAdditionalLatency} {1}

save_system {tagram.qsys}
