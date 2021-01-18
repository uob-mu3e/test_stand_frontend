#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join "./device.tcl" ]
source [ file join "./util/altera_ip.tcl" ]

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_xcvr_native_a10 6 32 ${refclk_freq_mhz} ${txrx_data_rate}
