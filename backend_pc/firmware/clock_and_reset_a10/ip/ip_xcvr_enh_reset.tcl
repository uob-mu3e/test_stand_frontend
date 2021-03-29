#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_reset_control ${xcvr_enh_channels} ${xcvr_enh_clk_mhz}
