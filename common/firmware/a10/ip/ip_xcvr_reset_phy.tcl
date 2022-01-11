#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_native_a10 ${xcvr_reset_channels} 8 ${xcvr_reset_refclk_mhz} ${xcvr_reset_rate_mbps}
