#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_native_a10 ${xcvr_enh_channels} 40 ${xcvr_enh_refclk_mhz} ${xcvr_enh_rate_mbps} -mode basic_enh

# bonded configuration
if { 1 } {
    set_instance_parameter_value xcvr_native_a10_0 {bonded_mode} {pma_pcs}
}
