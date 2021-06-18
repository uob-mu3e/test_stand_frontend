#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_native_a10 ${xcvr_sfp_channels} ${xcvr_sfp_width} ${xcvr_sfp_refclk_mhz} ${xcvr_sfp_rate_mbps}
