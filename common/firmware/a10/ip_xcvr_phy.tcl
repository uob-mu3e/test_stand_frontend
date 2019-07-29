#

package require qsys

proc add_altera_xcvr_native { name channels channel_width cdr_refclk_freq data_rate } {
    #set_project_property FABRIC_MODE value {NATIVE}

    # Instances and instance parameters
    add_instance ${name} altera_xcvr_native_a10
    set_instance_parameter_value ${name} {design_environment} {NATIVE}

    set_instance_parameter_value ${name} {support_mode} {user_mode}
    set_instance_parameter_value ${name} {protocol_mode} {basic_std}
    set_instance_parameter_value ${name} {pma_mode} {basic}
    set_instance_parameter_value ${name} {duplex_mode} {duplex}
    set_instance_parameter_value ${name} {channels} ${channels}
    set_instance_parameter_value ${name} {set_data_rate} ${data_rate}
    set_instance_parameter_value ${name} {enable_simple_interface} {1}
    set_instance_parameter_value ${name} {enable_split_interface} {0}

    set_instance_parameter_value ${name} {set_cdr_refclk_freq} ${cdr_refclk_freq}
    set_instance_parameter_value ${name} {rx_ppm_detect_threshold} {1000}

    set_instance_parameter_value ${name} {enable_port_rx_is_lockedtodata} {1}
    set_instance_parameter_value ${name} {enable_port_rx_is_lockedtoref} {1}

    set_instance_parameter_value ${name} {enable_port_rx_seriallpbken_tx} {1}
    set_instance_parameter_value ${name} {enable_port_rx_seriallpbken} {1}
    if { ${channel_width} == 8 } {
        set_instance_parameter_value ${name} {std_pcs_pma_width} {10}
    } \
    else {
        set_instance_parameter_value ${name} {std_pcs_pma_width} {20}
    }

    if { ${channel_width} == 32 } {
        set_instance_parameter_value ${name} {std_tx_byte_ser_mode} {Serialize x2}
        set_instance_parameter_value ${name} {std_rx_byte_deser_mode} {Deserialize x2}
    }

    set_instance_parameter_value ${name} {std_tx_8b10b_enable} {1}
    set_instance_parameter_value ${name} {std_rx_8b10b_enable} {1}

    set_instance_parameter_value ${name} {std_rx_word_aligner_mode} {synchronous state machine}
    set_instance_parameter_value ${name} {std_rx_word_aligner_pattern_len} {10}
    # word aligner pattern K28.5
    set_instance_parameter_value ${name} {std_rx_word_aligner_pattern} {0x283}

    set_instance_parameter_value ${name} {rcfg_enable} {1}
    set_instance_parameter_value ${name} {rcfg_shared} {1}
    set_instance_parameter_value ${name} {rcfg_separate_avmm_busy} {1}
    set_instance_parameter_value ${name} {set_capability_reg_enable} {1}
    set_instance_parameter_value ${name} {set_csr_soft_logic_enable} {1}

    # exported interfaces
    set_instance_property ${name} AUTO_EXPORT {true}
}

source {device.tcl}
create_system {ip_xcvr_phy}
add_altera_xcvr_native xcvr_native_a10_0 4 32 [ expr $refclk_freq * 1e-6 ] $txrx_data_rate
save_system {a10/ip_xcvr_phy.qsys}
