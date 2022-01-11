#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_reset_control ${xcvr_clk_channels} ${xcvr_clk_clk_mhz}
