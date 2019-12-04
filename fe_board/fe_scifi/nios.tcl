package require qsys

create_system {nios}
source {../fe/device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value spi numberOfSlaves 16

source {../fe/nios_avm.tcl}
source {../fe/nios_spi_si.tcl}

source {../firmware/FEB_common/nios_mscb_inc.tcl}

save_system {nios.qsys}
