#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_native_a10 6 32 ${refclk_freq_mhz} ${txrx_data_rate}
