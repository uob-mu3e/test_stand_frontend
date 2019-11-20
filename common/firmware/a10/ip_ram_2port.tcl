#

# ::add_ram_2port --
#
#   Add 2-port Intel FPGA IP (ram_2port) instance and auto export.
#   Default is 1rw (one read port and one write port) RAM.
#
# Arguments:
#   name        - name of the ip
#   width       - word width [bits]
#   words       - RAM size [words]
#   -2rw        - two read/write ports
#   -dc         - dual clock
#   -rdw $      - Read-During-Write (old - old data, new - new data)
#   -widthA $   - width of port A [bits]
#   -widthB $   - width of port B [bits]
#   -regA       - register outputs of port A
#   -regB       - register outputs of port B
#
proc add_ram_2port { name width words args } {
    set 2rw 0
    set dc 0
    set rdw 0
    set widthA ${width}
    set widthB ${width}
    set regA 0
    set regB 0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -2rw {
                set 2rw 1
            }
            -dc {
                set dc 1
            }
            -rdw { incr i
                set rdw [ lindex $args $i ]
            }
            -widthA { incr i
                set widthA [ lindex $args $i ]
            }
            -widthB { incr i
                set widthB [ lindex $args $i ]
            }
            -regA {
                set regA 1
            }
            -regB {
                set regB 1
            }
            default {
                send_message "Error" "\[add_ram_2port\] invalid argument '[ lindex $args $i ]'"
            }
        }
    }

    add_instance ${name} ram_2port

    set_instance_parameter_value ${name} {GUI_DATAA_WIDTH} ${width}
    set_instance_parameter_value ${name} {GUI_QA_WIDTH} ${widthA}
    set_instance_parameter_value ${name} {GUI_QB_WIDTH} ${widthB}

    set_instance_parameter_value ${name} {GUI_MEMSIZE_WORDS} ${words}

    # different data widths on different ports
    if { ${width} != ${widthA} || ${width} != ${widthB} } {
        set_instance_parameter_value ${name} {GUI_VAR_WIDTH} {1}
    }

    # two read/write ports
    if { ${2rw} == 1 } {
        set_instance_parameter_value ${name} {GUI_MODE} {1}
    }

    # dual clock
    if { ${2rw} == 0 && ${dc} == 1 } {
        # separate read and write clocks
        set_instance_parameter_value ${name} {GUI_CLOCK_TYPE} {1}
    }
    if { ${2rw} == 1 && ${dc} == 1 } {
        # custom clocks for A and B ports
        set_instance_parameter_value ${name} {GUI_CLOCK_TYPE} {4}
    }

    switch -- ${rdw} {
        0 {}
        old { set_instance_parameter_value ${name} {GUI_Q_PORT_MODE} {1} }
        new { set_instance_parameter_value ${name} {GUI_Q_PORT_MODE} {3} }
        default { send_message "Error" "\[add_ram_2port\] invalid RDW '[ lindex $args $i ]'" }
    }

    # output registers
    set_instance_parameter_value ${name} {GUI_READ_OUTPUT_QA} ${regA}
    set_instance_parameter_value ${name} {GUI_READ_OUTPUT_QB} ${regB}

    set_instance_property ${name} AUTO_EXPORT {true}
}
