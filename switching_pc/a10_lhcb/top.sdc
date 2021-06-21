#

# oscillator
create_clock -period "100.0 MHz" [get_ports CLK_A10_100MHZ_P]

# SI53344
create_clock -period "125.0 MHz" [get_ports A10_CUSTOM_CLK_P]

# SI5345_1
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_0]
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_1]
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_2]
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_3]

# SI5345_2
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_4]
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_5]
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_6]
create_clock -period "125.0 MHz" [get_ports A10_REFCLK_GBT_P_7]



derive_pll_clocks -create_base_clocks

derive_clock_uncertainty

# false paths
set_false_path -from {a10_block:a10_block|pcie_block:\generate_pcie0:e_pcie0_block|pcie_application:e_pcie_application|pcie_writeable_registers:e_pcie_writeable_registers|writeregs_r[19][*]} -to {*}