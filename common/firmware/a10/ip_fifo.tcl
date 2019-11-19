#

proc add_fifo { name width depth args } {
    set dcfifo 0
    set usedw 0
    set aclr 0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -dcfifo {
                set dcfifo 1
            }
            -usedw {
                set usedw 1
            }
            -aclr {
                set aclr 1
            }
            default {
                send_message "Error" "\[add_fifo\] invalid argument '[ lindex $args $i ]'"
            }
        }
    }

    add_instance ${name} fifo

    if { ${dcfifo} == 1 } {
        set_instance_parameter_value ${name} {GUI_Clock} {4}
    }

    set_instance_parameter_value ${name} {GUI_Width} ${width}
    set_instance_parameter_value ${name} {GUI_Depth} ${depth}

    # showahead mode
    set_instance_parameter_value ${name} {GUI_LegacyRREQ} {0}

    # usedw port
    set_instance_parameter_value ${name} {GUI_UsedW} ${usedw}

    # async clear port
    if { ${aclr} && ${dcfifo} == 0 } {
        set_instance_parameter_value ${name} {GUI_sc_aclr} {1}
    }
    if { ${aclr} && ${dcfifo} == 1 } {
        set_instance_parameter_value ${name} {GUI_dc_aclr} {1}
    }

    set_instance_property ${name} AUTO_EXPORT {true}
}
