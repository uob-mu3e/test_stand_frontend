package require qsys

create_system {ip_xcvr_reset}
source {device.tcl}

# Instances and instance parameters
add_instance xcvr_reset_control_0 altera_xcvr_reset_control
apply_preset xcvr_reset_control_0 "Arria 10 Default Settings"

set_instance_parameter_value xcvr_reset_control_0 {CHANNELS} {4}
set_instance_parameter_value xcvr_reset_control_0 {PLLS} {1}

set_instance_parameter_value xcvr_reset_control_0 {SYS_CLK_IN_MHZ} [ expr $refclk_freq * 1e-6 ]
set_instance_parameter_value xcvr_reset_control_0 {gui_pll_cal_busy} {1}
set_instance_parameter_value xcvr_reset_control_0 {RX_PER_CHANNEL} {1}

# exported interfaces
set_instance_property xcvr_reset_control_0 AUTO_EXPORT {true}

save_system {ip/ip_xcvr_reset.qsys}
