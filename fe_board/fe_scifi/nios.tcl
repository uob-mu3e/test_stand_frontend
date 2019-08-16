package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 16

source {../fe/nios_avm.tcl}

source {../firmware/FEB_common/nios_mscb_inc.tcl}

source {../fe_malibu/nios_irq_bridge.tcl}
add_irq_bridge irq_bridge_0 4 cpu avm_clk

save_system {nios.qsys}
