#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_reset_control ${xcvr_enh_channels} ${xcvr_enh_clk_mhz}
