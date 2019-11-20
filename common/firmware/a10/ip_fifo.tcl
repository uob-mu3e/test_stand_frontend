#

# ::add_fifo --
#
#   Add FIFO Intel FPGA IP (fifo) instance and auto export.
#
# Arguments:
#   name        - name of the ip
#   width       - fifo width [bits]
#   depth       - fifo depth [words]
#   -dc         - dual clock
#   -usedw      - add usedw port (number of words in the fifo)
#   -aclr       - add aclr port (asynchronous clear)
#
proc ::add_altera_fifo { name width depth args } {
    set dc 0
    set usedw 0
    set aclr 0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -dc {
                set dc 1
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

    if { ${dc} == 1 } {
        set_instance_parameter_value ${name} {GUI_Clock} {4}
    }

    set_instance_parameter_value ${name} {GUI_Width} ${width}
    set_instance_parameter_value ${name} {GUI_Depth} ${depth}

    # showahead mode
    set_instance_parameter_value ${name} {GUI_LegacyRREQ} {0}

    # usedw port
    set_instance_parameter_value ${name} {GUI_UsedW} ${usedw}

    # async clear port
    if { ${aclr} && ${dc} == 0 } {
        set_instance_parameter_value ${name} {GUI_sc_aclr} {1}
    }
    if { ${aclr} && ${dc} == 1 } {
        set_instance_parameter_value ${name} {GUI_dc_aclr} {1}
    }

    set_instance_property ${name} AUTO_EXPORT {true}
}
