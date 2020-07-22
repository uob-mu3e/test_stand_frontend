#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join $dir0 "../device.tcl" ]
source [ file join $dir0 "../util/altera_ip.tcl" ]



set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_modular_adc { 1 2 3 4 5 6 7 8 tsd } -seq_order { 17 1 2 3 4 5 6 7 }
save_system [ file join $dir0 "$name.qsys" ]
