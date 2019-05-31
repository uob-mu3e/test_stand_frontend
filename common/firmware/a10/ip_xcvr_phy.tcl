package require qsys

create_system {ip_xcvr_phy}
source {device.tcl}
#set_project_property FABRIC_MODE value {NATIVE}

# Instances and instance parameters
add_instance xcvr_native_a10_0 altera_xcvr_native_a10
set_instance_parameter_value xcvr_native_a10_0 {design_environment} {NATIVE}

set_instance_parameter_value xcvr_native_a10_0 {support_mode} {user_mode}
set_instance_parameter_value xcvr_native_a10_0 {protocol_mode} {basic_std}
set_instance_parameter_value xcvr_native_a10_0 {pma_mode} {basic}
set_instance_parameter_value xcvr_native_a10_0 {duplex_mode} {duplex}
set_instance_parameter_value xcvr_native_a10_0 {channels} {4}
set_instance_parameter_value xcvr_native_a10_0 {set_data_rate} $txrx_data_rate
set_instance_parameter_value xcvr_native_a10_0 {enable_simple_interface} {1}
set_instance_parameter_value xcvr_native_a10_0 {enable_split_interface} {0}

set_instance_parameter_value xcvr_native_a10_0 {set_cdr_refclk_freq} [ expr $refclk_freq * 1e-6 ]
set_instance_parameter_value xcvr_native_a10_0 {rx_ppm_detect_threshold} {1000}

set_instance_parameter_value xcvr_native_a10_0 {enable_port_rx_is_lockedtodata} {1}
set_instance_parameter_value xcvr_native_a10_0 {enable_port_rx_is_lockedtoref} {1}

set_instance_parameter_value xcvr_native_a10_0 {enable_port_rx_seriallpbken_tx} {1}
set_instance_parameter_value xcvr_native_a10_0 {enable_port_rx_seriallpbken} {1}
set_instance_parameter_value xcvr_native_a10_0 {std_pcs_pma_width} {20}

set_instance_parameter_value xcvr_native_a10_0 {std_tx_byte_ser_mode} {Serialize x2}
set_instance_parameter_value xcvr_native_a10_0 {std_rx_byte_deser_mode} {Deserialize x2}

set_instance_parameter_value xcvr_native_a10_0 {std_tx_8b10b_enable} {1}
set_instance_parameter_value xcvr_native_a10_0 {std_rx_8b10b_enable} {1}

set_instance_parameter_value xcvr_native_a10_0 {std_rx_word_aligner_mode} {synchronous state machine}
set_instance_parameter_value xcvr_native_a10_0 {std_rx_word_aligner_pattern_len} {10}
# word aligner pattern K28.5
set_instance_parameter_value xcvr_native_a10_0 {std_rx_word_aligner_pattern} {0x283}
#set_instance_parameter_value xcvr_native_a10_0 {std_rx_word_aligner_pattern} {0x383}

set_instance_parameter_value xcvr_native_a10_0 {rcfg_enable} {1}
set_instance_parameter_value xcvr_native_a10_0 {rcfg_shared} {1}
set_instance_parameter_value xcvr_native_a10_0 {rcfg_separate_avmm_busy} {1}
set_instance_parameter_value xcvr_native_a10_0 {set_capability_reg_enable} {1}
set_instance_parameter_value xcvr_native_a10_0 {set_csr_soft_logic_enable} {1}

# exported interfaces
set_instance_property xcvr_native_a10_0 AUTO_EXPORT {true}

save_system {ip/ip_xcvr_phy.qsys}
