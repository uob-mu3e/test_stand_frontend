#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"



set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_modular_adc { 1 2 3 4 5 6 7 8 tsd } -seq_order { 17 1 2 3 4 5 6 7 }
