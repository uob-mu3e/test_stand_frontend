package require qsys

create_system {ip_iopll_125}
source {device.tcl}

add_instance iopll_0 altera_iopll 18.0
set_instance_parameter_value iopll_0 {gui_reference_clock_frequency} {50.0}
set_instance_parameter_value iopll_0 {gui_output_clock_frequency0} {125.0}
set_instance_parameter_value iopll_0 {gui_pll_auto_reset} {1}
set_instance_parameter_value iopll_0 {gui_use_locked} {0}

set_instance_property iopll_0 AUTO_EXPORT {true}

save_system {ip/ip_iopll_125.qsys}
