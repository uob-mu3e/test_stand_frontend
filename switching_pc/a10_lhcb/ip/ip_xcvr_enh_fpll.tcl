#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_fpll_a10 ${xcvr_enh_refclk_mhz} [ expr ${xcvr_enh_data_mbps} / 2 ]
