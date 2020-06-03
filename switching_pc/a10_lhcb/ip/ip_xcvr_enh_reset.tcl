#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join $dir0 "../device.tcl" ]
source [ file join $dir0 "../util/altera_ip.tcl" ]

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_reset_control ${xcvr_enh_channels} ${xcvr_enh_clk_mhz}
save_system [ file join $dir0 "$name.qsys" ]
