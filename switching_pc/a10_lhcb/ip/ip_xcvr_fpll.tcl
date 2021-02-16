#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_fpll_a10 ${refclk_freq_mhz} [ expr ${txrx_data_rate} / 2 ]
