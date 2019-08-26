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



save_system {nios.qsys}
