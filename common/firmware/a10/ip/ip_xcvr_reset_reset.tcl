#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_reset_control ${xcvr_reset_channels} ${xcvr_reset_clk_mhz}
