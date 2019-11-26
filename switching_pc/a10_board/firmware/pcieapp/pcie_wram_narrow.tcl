#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join $dir0 "../device.tcl" ]
source [ file join $dir0 "../util/altera_ip.tcl" ]

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_ram_2port 32 65536 -2rw -dc
save_system [ file join $dir0 "$name.qsys" ]
