#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join "./device.tcl" ]
source [ file join "./util/altera_ip.tcl" ]

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_reset_control 4 ${refclk_freq_mhz}
save_system [ file join $dir0 "$name.qsys" ]
