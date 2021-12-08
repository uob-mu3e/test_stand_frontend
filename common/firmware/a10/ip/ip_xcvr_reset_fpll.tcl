#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_fpll_a10 ${xcvr_reset_refclk_mhz} [ expr ${xcvr_reset_rate_mbps} / 2 ]
