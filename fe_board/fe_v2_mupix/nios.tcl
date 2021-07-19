#

source {device.tcl}

source {util/nios_base.tcl}
set_instance_parameter_value ram {memorySize} {0x00015000}
set_instance_parameter_value spi numberOfSlaves 16

proc call_python {} {
    set output [exec python3 ../../common/include/makeheader.py ../../common/firmware/registers/mupix_registers.vhd ../../common/firmware/include/mupix_registers.h]
    puts $output
}

call_python

source {../fe/nios_avm.tcl}
source {../fe/nios_spi_si.tcl}
source {../fe/nios_tmp.tcl}
