#

nios_base.add_clock_source avm_clk 156250000 -reset_export avm_reset

nios_base.add_irq_bridge irq_bridge 4 -clk avm_clk

nios_base.export_avm avm_sc 16 0x70080000 -readLatency 1 -clk avm_clk

nios_base.export_avm avm_qsfp 14 0x70010000
nios_base.export_avm avm_pod 14 0x70020000
