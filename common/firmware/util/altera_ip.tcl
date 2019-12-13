#

# ::add_fifo --
#
#   Add FIFO Intel FPGA IP (fifo) instance and auto export.
#
# Arguments:
#   width       - fifo width [bits]
#   depth       - fifo depth [words]
#   -dc         - dual clock
#   -usedw      - add usedw port (number of words in the fifo)
#   -aclr       - add aclr port (asynchronous clear)
#   -name $     - instance name
#
proc ::add_fifo { width depth args } {
    set name fifo_0
    set dc 0
    set usedw 0
    set aclr 0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -name { incr i
                set name [ lindex $args $i ]
            }
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

# ::add_ram_2port --
#
#   Add 2-port Intel FPGA IP (ram_2port) instance and auto export.
#   Default is 1rw (one read port and one write port) RAM.
#
# Arguments:
#   width       - word width [bits]
#   words       - RAM size [words]
#   -2rw        - two read/write ports
#   -dc         - dual clock
#   -rdw $      - Read-During-Write (old - old data, new - new data)
#   -widthA $   - width of port A [bits]
#   -widthB $   - width of port B [bits]
#   -regA       - register outputs of port A
#   -regB       - register outputs of port B
#   -name $     - instance name
#
proc ::add_ram_2port { width words args } {
    set name ram_2port_0
    set 2rw 0
    set dc 0
    set rdw 0
    set widthA ${width}
    set widthB ${width}
    set regA 0
    set regB 0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -name { incr i
                set name [ lindex $args $i ]
            }
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

# ::add_altclkctrl --
#
#   Add Clock Control Block IP (altclkctrl) instance and auto export.
#
# Arguments:
#   n           - number of inputs
#   -name $     - instance name
#
proc ::add_altclkctrl { n args } {
    set name altclkctrl_0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -name { incr i
                set name [ lindex $args $i ]
            }
            default {
                send_message "Error" "\[add_altclkctrl\] invalid argument '[ lindex $args $i ]'"
            }
        }
    }

    add_instance ${name} altclkctrl

    set_instance_parameter_value ${name} {CLOCK_TYPE} {0}

    set_instance_parameter_value ${name} {NUMBER_OF_CLOCKS} ${n}

    set_instance_property ${name} AUTO_EXPORT {true}
}

# ::add_altclkctrl --
#
#   Add IOPLL Intel FPGA IP (altera_iopll) instance and auto export.
#
# Arguments:
#   refclk      - reference clock frequency [MHz]
#   outclk      -
#   -locked     - add locked port
#   -name $     - instance name
#
proc ::add_altera_iopll { refclk outclk args } {
    set name iopll_0
    set locked 0
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -name { incr i
                set name [ lindex $args $i ]
            }
            -locked {
                set locked 1
            }
            default {
                send_message "Error" "\[add_altclkctrl\] invalid argument '[ lindex $args $i ]'"
            }
        }
    }

    add_instance ${name} altera_iopll

    set_instance_parameter_value ${name} {gui_reference_clock_frequency} ${refclk}

    set_instance_parameter_value ${name} {gui_output_clock_frequency0} ${outclk}

    set_instance_parameter_value ${name} {gui_use_locked} ${locked}

    set_instance_property ${name} AUTO_EXPORT {true}
}

# ::add_modular_adc --
#
#   Add Modular ADC Core instance and auto export.
#
# Arguments:
#   channels    - list of active channels
#   -seq_order  - channel acquisition sequence
#   -name $     - instance name
#
proc ::add_modular_adc { channels args } {
    set name modular_adc_0
    set seq_order ""
    for { set i 0 } { $i < [ llength $args ] } { incr i } {
        switch -- [ lindex $args $i ] {
            -name { incr i
                set name [ lindex $args $i ]
            }
            -seq_order { incr i
                set seq_order [ lindex $args $i ]
            }
            default {
                send_message "Error" "\[add_modular_adc\] invalid argument '[ lindex $args $i ]'"
            }
        }
    }

    add_instance ${name} altera_modular_adc

    set_instance_parameter_value ${name} {CORE_VAR} {3}

    foreach channel $channels {
        if { [ string equal $channel tsd ] } {
            # temperature sensing diode
            set_instance_parameter_value ${name} {use_tsd} {1}
        } \
        else {
            set_instance_parameter_value ${name} use_ch$channel {1}
        }
    }

    set n 0
    foreach slot $seq_order {
        incr n
    }
    if { $n > 0 } {
        set_instance_parameter_value ${name} {seq_order_length} $n
        set i 0
        foreach slot $seq_order {
            incr i
            set_instance_parameter_value ${name} seq_order_slot_$i $slot
        }
    }

    set_instance_property ${name} AUTO_EXPORT {true}
}
