#

package require qsys

source [ file join "./util/altera_ip.tcl" ]

source {device.tcl}
create_system {ip_xcvr_phy}
add_altera_xcvr_native_a10 4 32 [ expr $refclk_freq * 1e-6 ] $txrx_data_rate
save_system {a10/ip_xcvr_phy.qsys}
