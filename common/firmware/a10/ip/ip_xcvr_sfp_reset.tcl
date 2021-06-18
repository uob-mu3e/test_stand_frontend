#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_reset_control ${xcvr_sfp_channels} ${xcvr_sfp_clk_mhz}
