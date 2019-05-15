package require qsys

create_system {nios}
source {device.tcl}



# clock
add_instance clk_50 clock_source
set_instance_parameter_value clk_50 {clockFrequency} [ expr $refclk_freq * 1000000 ]

# cpu
add_instance cpu altera_nios2_gen2
set_instance_parameter_value cpu {resetSlave} {flash.uas}
set_instance_parameter_value cpu {resetOffset} {0x05E40000}
set_instance_parameter_value cpu {exceptionSlave} {ram.s1}

# ram
add_instance ram altera_avalon_onchip_memory2
set_instance_parameter_value ram {memorySize} {0x00080000}
set_instance_parameter_value ram {initMemContent} {0}

# flash
add_instance flash a10_flash1616

# jtag master
add_instance jtag_master altera_jtag_avalon_master



add_connection clk_50.clk cpu.clk
add_connection clk_50.clk ram.clk1
add_connection clk_50.clk flash.clk
add_connection clk_50.clk jtag_master.clk

add_connection clk_50.clk_reset cpu.reset
add_connection clk_50.clk_reset ram.reset1
add_connection clk_50.clk_reset flash.reset
add_connection clk_50.clk_reset jtag_master.clk_reset

add_connection                 cpu.data_master flash.uas
set_connection_parameter_value cpu.data_master/flash.uas                   baseAddress {0x00000000}
add_connection                 cpu.instruction_master flash.uas
set_connection_parameter_value cpu.instruction_master/flash.uas            baseAddress {0x00000000}
add_connection                 cpu.data_master ram.s1
set_connection_parameter_value cpu.data_master/ram.s1                      baseAddress {0x10000000}
add_connection                 cpu.instruction_master ram.s1
set_connection_parameter_value cpu.instruction_master/ram.s1               baseAddress {0x10000000}
add_connection                 cpu.data_master cpu.debug_mem_slave
set_connection_parameter_value cpu.data_master/cpu.debug_mem_slave         baseAddress {0x70000000}
add_connection                 cpu.instruction_master cpu.debug_mem_slave
set_connection_parameter_value cpu.instruction_master/cpu.debug_mem_slave  baseAddress {0x70000000}



add_connection jtag_master.master flash.uas
add_connection jtag_master.master ram.s1
add_connection jtag_master.master cpu.debug_mem_slave
add_connection cpu.debug_reset_request cpu.reset
add_connection cpu.debug_reset_request ram.reset1
add_connection cpu.debug_reset_request flash.reset



# exported interfaces
add_interface clk clock sink
set_interface_property clk EXPORT_OF clk_50.clk_in
add_interface reset reset sink
set_interface_property reset EXPORT_OF clk_50.clk_in_reset
add_interface flash conduit end
set_interface_property flash EXPORT_OF flash.out


# uart, timers, i2c
if 1 {
    add_instance sysid altera_avalon_sysid_qsys

    add_instance jtag_uart altera_avalon_jtag_uart

    add_instance timer altera_avalon_timer
    apply_preset timer "Simple periodic interrupt"
    set_instance_parameter_value timer {period} {1}
    set_instance_parameter_value timer {periodUnits} {MSEC}

    add_instance timer_ts altera_avalon_timer
    apply_preset timer_ts "Full-featured"

    add_instance i2c altera_avalon_i2c

    foreach { name clk reset mm addr } {
        sysid     clk   reset      control_slave     0x0000
        jtag_uart clk   reset      avalon_jtag_slave 0x0010
        timer     clk   reset      s1                0x0100
        timer_ts  clk   reset      s1                0x0140
        i2c       clock reset_sink csr               0x0200
    } {
        add_connection clk_50.clk       $name.$clk
        add_connection clk_50.clk_reset $name.$reset
        add_connection                 cpu.data_master $name.$mm
        set_connection_parameter_value cpu.data_master/$name.$mm baseAddress [ expr 0x700F0000 + $addr ]
        add_connection cpu.debug_reset_request $name.$reset
    }

    # IRQ assignments
    foreach { name irq } {
        jtag_uart.irq 3
        timer.irq 0
        i2c.interrupt_sender 4
    } {
        add_connection cpu.irq $name
        set_connection_parameter_value cpu.irq/$name irqNumber $irq
    }

    add_interface i2c conduit end
    set_interface_property i2c EXPORTOF i2c.i2c_serial
}



save_system {nios.qsys}
