#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join $dir0 "../device.tcl" ]
source [ file join $dir0 "../util/altera_ip.tcl" ]

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_fpll_a10 ${xcvr_enh_refclk_mhz} [ expr ${xcvr_enh_data_mbps} / 2 ]
save_system [ file join $dir0 "$name.qsys" ]