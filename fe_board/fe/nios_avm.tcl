#

nios_base.export_avm avm_qsfp 14 0x70010000
nios_base.export_avm avm_pod 14 0x70020000

nios_base.add_clock_source avm_clk 156250000 -reset_export avm_reset

nios_base.export_avm avm_sc 16 0x70080000 -readLatency 1

add_connection avm_clk.clk       avm_sc.clk
add_connection avm_clk.clk_reset avm_sc.reset
