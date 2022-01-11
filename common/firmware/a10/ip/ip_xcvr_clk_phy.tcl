#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_native_a10 ${xcvr_clk_channels} ${xcvr_clk_width} ${xcvr_clk_refclk_mhz} ${xcvr_clk_rate_mbps} -mode basic_enh
