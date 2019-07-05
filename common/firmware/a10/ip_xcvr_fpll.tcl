package require qsys

create_system {ip_xcvr_fpll}
source {device.tcl}

# Instances and instance parameters
add_instance xcvr_fpll_a10_0 altera_xcvr_fpll_a10

# fpll mode Transceiver
set_instance_parameter_value xcvr_fpll_a10_0 {gui_fpll_mode} {2}
# protocol mode Basic
set_instance_parameter_value xcvr_fpll_a10_0 {gui_hssi_prot_mode} {0}

set_instance_parameter_value xcvr_fpll_a10_0 {gui_desired_refclk_frequency} [ expr $refclk_freq * 1e-6 ]
set_instance_parameter_value xcvr_fpll_a10_0 {gui_actual_refclk_frequency} [ expr $refclk_freq * 1e-6 ]

# bandwidth High
set_instance_parameter_value xcvr_fpll_a10_0 {gui_bw_sel} {high}
# operation mode Direct
set_instance_parameter_value xcvr_fpll_a10_0 {gui_operation_mode} {0}

set_instance_parameter_value xcvr_fpll_a10_0 {gui_hssi_output_clock_frequency} [ expr $txrx_data_rate / 2 ]

set_instance_parameter_value xcvr_fpll_a10_0 {enable_pll_reconfig} {1}
set_instance_parameter_value xcvr_fpll_a10_0 {rcfg_separate_avmm_busy} {1}
set_instance_parameter_value xcvr_fpll_a10_0 {set_capability_reg_enable} {1}
set_instance_parameter_value xcvr_fpll_a10_0 {set_csr_soft_logic_enable} {1}

# exported interfaces
set_instance_property xcvr_fpll_a10_0 AUTO_EXPORT {true}

save_system {ip/ip_xcvr_fpll.qsys}
