#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_reset_control 6 ${refclk_freq_mhz}
