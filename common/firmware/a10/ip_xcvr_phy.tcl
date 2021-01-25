#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_native_a10 4 32 ${refclk_freq_mhz} ${txrx_data_rate}
