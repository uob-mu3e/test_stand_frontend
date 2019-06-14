package require qsys

create_system {ip_clkctrl}
source {device.tcl}

add_instance altclkctrl_0 altclkctrl
set_instance_parameter_value altclkctrl_0 {CLOCK_TYPE} {0}
set_instance_parameter_value altclkctrl_0 {NUMBER_OF_CLOCKS} {1}

set_instance_property altclkctrl_0 AUTO_EXPORT {true}

save_system {ip/ip_clkctrl.qsys}
