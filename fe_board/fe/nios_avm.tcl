#

add_instance                 avm_clk clock_source
set_instance_parameter_value avm_clk {clockFrequency} {156250000}
set_instance_parameter_value avm_clk {resetSynchronousEdges} {DEASSERT}

add_interface          avm_clk clock sink
set_interface_property avm_clk EXPORT_OF avm_clk.clk_in
add_interface          avm_reset reset sink
set_interface_property avm_reset EXPORT_OF avm_clk.clk_in_reset

nios_base.export_avm avm_qsfp 14 0x70010000 -addressUnits 32
nios_base.export_avm avm_pod 14 0x70020000 -addressUnits 32
nios_base.export_avm avm_sc 16 0x70080000 -addressUnits 32 -readLatency 1

add_connection avm_clk.clk       avm_sc.clk
add_connection avm_clk.clk_reset avm_sc.reset
