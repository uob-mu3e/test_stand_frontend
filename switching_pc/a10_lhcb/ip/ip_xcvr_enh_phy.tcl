#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_native_a10 ${xcvr_enh_channels} 40 ${xcvr_enh_refclk_mhz} ${xcvr_enh_data_mbps} -mode basic_enh
