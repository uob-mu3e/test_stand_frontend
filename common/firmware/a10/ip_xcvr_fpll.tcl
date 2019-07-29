#

package require qsys

proc add_altera_xcvr_fpll { name refclk_frequency output_clock_frequency } {
    add_instance ${name} altera_xcvr_fpll_a10

    # fpll mode Transceiver
    set_instance_parameter_value ${name} {gui_fpll_mode} {2}
    # protocol mode Basic
    set_instance_parameter_value ${name} {gui_hssi_prot_mode} {0}

    set_instance_parameter_value ${name} {gui_desired_refclk_frequency} ${refclk_frequency}
    set_instance_parameter_value ${name} {gui_actual_refclk_frequency} ${refclk_frequency}

    # bandwidth High
    set_instance_parameter_value ${name} {gui_bw_sel} {high}
    # operation mode Direct
    set_instance_parameter_value ${name} {gui_operation_mode} {0}

    set_instance_parameter_value ${name} {gui_hssi_output_clock_frequency} ${output_clock_frequency}

    set_instance_parameter_value ${name} {enable_pll_reconfig} {1}
    set_instance_parameter_value ${name} {rcfg_separate_avmm_busy} {1}
    set_instance_parameter_value ${name} {set_capability_reg_enable} {1}
    set_instance_parameter_value ${name} {set_csr_soft_logic_enable} {1}

    set_instance_property ${name} AUTO_EXPORT {true}
}

source {device.tcl}
create_system {ip_xcvr_fpll}
add_altera_xcvr_fpll altera_xcvr_fpll_a10 [ expr $refclk_freq * 1e-6 ] [ expr $txrx_data_rate / 2 ]
save_system {a10/ip_xcvr_fpll.qsys}
