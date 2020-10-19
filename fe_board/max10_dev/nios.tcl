package require qsys

create_system {nios}
source {device.tcl}

source {util/nios_base_max10.tcl}
set_instance_parameter_value ram {memorySize} {0x00008000}

nios_base.add_pio i_pio 32 Input 0x700F0320

source "nios_adc.tcl"
source "nios_ufm.tcl"



save_system {nios.qsys}
