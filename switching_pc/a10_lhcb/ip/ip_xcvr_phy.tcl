#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_native_a10 ${xcvr_channels} 32 ${xcvr_refclk_mhz} ${xcvr_rate_mbps}
