package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00008000}



add_instance adc altera_modular_adc
set_instance_parameter_value adc {use_tsd} {1}
# Conversion Sequence Length
set_instance_parameter_value adc {seq_order_length} {1}
# Conversion Sequence Channels
set_instance_parameter_value adc {seq_order_slot_1} {17}

set_interface_property adc_pll_clock EXPORT_OF adc.adc_pll_clock
set_interface_property adc_pll_locked EXPORT_OF adc.adc_pll_locked



nios_base.connect adc clock reset_sink sequencer_csr 0x700F0380
nios_base.connect adc ""    ""         sample_store_csr 0x700F0400

foreach { name irq } {
    adc.sample_store_irq 1
} {
    add_connection cpu.irq $name
    set_connection_parameter_value cpu.irq/$name irqNumber $irq
}



add_instance flash altera_onchip_flash
set_instance_parameter_value flash {CLOCK_FREQUENCY} [ expr $nios_freq / 1000000 ]
set_instance_parameter_value flash {DATA_INTERFACE} {Parallel}

nios_base.connect flash clk nreset data 0x00000000
nios_base.connect flash ""    ""   csr 0x700F00F0



save_system {nios.qsys}
