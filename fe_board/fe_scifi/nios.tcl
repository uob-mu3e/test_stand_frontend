package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 5

source {../fe/nios_avm.tcl}

save_system {nios.qsys}
