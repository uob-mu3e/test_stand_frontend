#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_fpll_a10 ${refclk_freq_mhz} [ expr ${txrx_data_rate} / 2 ]
