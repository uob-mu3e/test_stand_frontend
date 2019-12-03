# qsys scripting (.tcl) file for clk_ctrl
package require -exact qsys 16.0

create_system {clk_ctrl}

set_project_property DEVICE_FAMILY {Stratix IV}
set_project_property DEVICE {EP4SGX70HF35C3}
set_project_property HIDE_FROM_IP_CATALOG {true}

# Instances and instance parameters
# (disabled instances are intentionally culled)
add_instance altclkctrl_0 altclkctrl 18.1
set_instance_parameter_value altclkctrl_0 {CLOCK_TYPE} {0}
set_instance_parameter_value altclkctrl_0 {ENA_REGISTER_MODE} {1}
set_instance_parameter_value altclkctrl_0 {GUI_USE_ENA} {0}
set_instance_parameter_value altclkctrl_0 {NUMBER_OF_CLOCKS} {2}
set_instance_parameter_value altclkctrl_0 {USE_GLITCH_FREE_SWITCH_OVER_IMPLEMENTATION} {0}

# exported interfaces
set_instance_property altclkctrl_0 AUTO_EXPORT {true}

# interconnect requirements
set_interconnect_requirement {$system} {qsys_mm.clockCrossingAdapter} {HANDSHAKE}
set_interconnect_requirement {$system} {qsys_mm.enableEccProtection} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.insertDefaultSlave} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.maxAdditionalLatency} {1}

save_system {clk_ctrl.qsys}
