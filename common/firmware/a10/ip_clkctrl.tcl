package require qsys

proc add_altclkctrl { name NUMBER_OF_CLOCKS } {
    add_instance ${name} altclkctrl

    set_instance_parameter_value ${name} {CLOCK_TYPE} {0}
    set_instance_parameter_value ${name} {NUMBER_OF_CLOCKS} ${NUMBER_OF_CLOCKS}

    set_instance_property ${name} AUTO_EXPORT {true}
}

source {device.tcl}
create_system {ip_clkctrl}
add_altclkctrl altclkctrl_0 1
save_system {a10/ip_clkctrl.qsys}
