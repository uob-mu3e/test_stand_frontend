#

package require qsys

source [ file join "./util/altera_ip.tcl" ]

source {device.tcl}
create_system {ip_xcvr_fpll}
add_altera_xcvr_fpll_a10 [ expr $refclk_freq * 1e-6 ] [ expr $txrx_data_rate / 2 ]
save_system {a10/ip_xcvr_fpll.qsys}
