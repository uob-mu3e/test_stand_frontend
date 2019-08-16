#

proc add_irq_bridge { name width cpu clk } {
    add_instance ${name} altera_irq_bridge

    # signal width
    set_instance_parameter_value ${name} {IRQ_WIDTH} ${width}
    # signal polarity
    set_instance_parameter_value ${name} {IRQ_N} {0}

    for { set i 0 } { $i < $width } { incr i } {
        add_connection                 ${cpu}.irq ${name}.sender${i}_irq
        set_connection_parameter_value ${cpu}.irq/${name}.sender${i}_irq irqNumber [ expr 16 + $i ]
    }

    add_connection ${clk}.clk ${name}.clk
    add_connection ${clk}.clk_reset ${name}.clk_reset

    add_interface irq interrupt receiver
    set_interface_property irq EXPORT_OF ${name}.receiver_irq
}
