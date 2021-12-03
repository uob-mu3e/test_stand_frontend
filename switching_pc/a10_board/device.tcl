#

set_project_property DEVICE_FAMILY {Arria 10}
set_project_property DEVICE {10AX115N2F45E1SG}

set nios_clk_mhz 50.0

set xcvr_clk_mhz ${nios_clk_mhz}
set xcvr_refclk_mhz 125.0
set xcvr_rate_mbps 6250
set xcvr_channels 4

set xcvr_enh_clk_mhz ${nios_clk_mhz}
set xcvr_enh_refclk_mhz 125.0
set xcvr_enh_rate_mbps 10000
set xcvr_enh_channels 4

set xcvr_reset_clk_mhz ${nios_clk_mhz}
set xcvr_reset_refclk_mhz 125.0
set xcvr_reset_rate_mbps 1250
set xcvr_reset_channels 4

set xcvr_sfp_clk_mhz ${nios_clk_mhz}
set xcvr_sfp_refclk_mhz 125.0
set xcvr_sfp_rate_mbps 1250
set xcvr_sfp_channels 2
set xcvr_sfp_width 8
