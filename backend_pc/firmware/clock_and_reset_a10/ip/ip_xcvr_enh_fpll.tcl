#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_xcvr_fpll_a10 ${xcvr_enh_refclk_mhz} [ expr ${xcvr_enh_rate_mbps} / 2 ]

# bonded configuration
if { 1 } {
    set_instance_parameter_value xcvr_fpll_a10_0 {enable_mcgb} {1}
    set_instance_parameter_value xcvr_fpll_a10_0 {enable_bonding_clks} {1}
    set_instance_parameter_value xcvr_fpll_a10_0 {pma_width} {40}
}
