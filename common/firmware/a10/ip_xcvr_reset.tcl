#

package require qsys

source [ file join "./util/altera_ip.tcl" ]

source {device.tcl}
create_system {ip_xcvr_reset}
add_altera_xcvr_reset_control 4 [ expr $refclk_freq * 1e-6 ]
save_system {a10/ip_xcvr_reset.qsys}
