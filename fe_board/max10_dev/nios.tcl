package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00008000}



add_instance adc altera_modular_adc
set_instance_parameter_value adc use_tsd {1}
# Conversion Sequence Length
set_instance_parameter_value adc seq_order_length {1}
# Conversion Sequence Channels
set_instance_parameter_value adc {seq_order_slot_1} {17}

set_interface_property adc_clock EXPORT_OF adc.adc_pll_clock
set_interface_property adc_locked EXPORT_OF adc.adc_pll_locked



    add_instance led_io altera_avalon_pio
    set_instance_parameter_value led_io {direction} {Output}
    set_instance_parameter_value led_io {width} {3}
    set_instance_parameter_value led_io {bitModifyingOutReg} {3}

    add_interface led_io conduit end
    set_interface_property led_io EXPORT_OF led_io.external_connection



    add_instance sw_io altera_avalon_pio
    set_instance_parameter_value sw_io {direction} {Input}
    set_instance_parameter_value sw_io {width} {3}

    add_interface sw_io conduit end
    set_interface_property sw_io EXPORT_OF sw_io.external_connection



    foreach { name clk reset avalon addr } {
        led_io          clk   reset      s1                0x0320
        sw_io           clk   reset      s1                0x0340
        adc             clock reset_sink sequencer_csr     0x0380
        adc             ""    ""         sample_store_csr  0x0400
    } {
        if { [string length $clk] > 0 } {
            add_connection clk.clk $name.$clk
        }
        if { [string length $reset] > 0 } {
            add_connection clk.clk_reset $name.$reset
            add_connection cpu.debug_reset_request $name.$reset
        }
        add_connection                 cpu.data_master $name.$avalon
        set_connection_parameter_value cpu.data_master/$name.$avalon baseAddress [ expr 0x700F0000 + $addr ]
    }



    foreach { name irq } {
        adc.sample_store_irq 1
    } {
        add_connection cpu.irq $name
        set_connection_parameter_value cpu.irq/$name irqNumber $irq
    }



save_system {nios.qsys}
