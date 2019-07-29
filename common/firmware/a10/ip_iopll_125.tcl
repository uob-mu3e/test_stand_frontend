package require qsys

proc add_iopll { name reference_clock_frequency output_clock_frequency } {
    add_instance ${name} altera_iopll 18.0

    set_instance_parameter_value ${name} {gui_reference_clock_frequency} ${reference_clock_frequency}
    set_instance_parameter_value ${name} {gui_output_clock_frequency0} ${output_clock_frequency}
    set_instance_parameter_value ${name} {gui_pll_auto_reset} {1}
    set_instance_parameter_value ${name} {gui_use_locked} {0}

    set_instance_property ${name} AUTO_EXPORT {true}
}

source {device.tcl}
create_system {ip_iopll_125}
add_iopll iopll_0 50.0 125.0
save_system {a10/ip_iopll_125.qsys}
