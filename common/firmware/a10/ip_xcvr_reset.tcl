#

package require qsys

proc add_altera_xcvr_reset_control { name CHANNELS SYS_CLK_IN_MHZ } {
    add_instance $name altera_xcvr_reset_control
    apply_preset $name "Arria 10 Default Settings"

    foreach { parameter value } [ list      \
        CHANNELS            $CHANNELS       \
        PLLS                1               \
        SYS_CLK_IN_MHZ      $SYS_CLK_IN_MHZ \
        gui_pll_cal_busy    1               \
        RX_PER_CHANNEL      1               \
    ] {
        set_instance_parameter_value $name $parameter $value
    }

    set_instance_property $name AUTO_EXPORT {true}
}

source {device.tcl}
create_system {ip_xcvr_reset}
add_altera_xcvr_reset_control xcvr_reset_control_0 4 [ expr $refclk_freq * 1e-6 ]
save_system {ip/ip_xcvr_reset.qsys}
