#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_fpll_a10 ${xcvr_clk_refclk_mhz} [ expr ${xcvr_clk_rate_mbps} / 2 ]
