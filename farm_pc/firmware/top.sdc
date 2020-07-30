#

create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2J]
create_clock -period "266.667000 MHz" [get_ports DDR3A_REFCLK_p]
create_clock -period "266.667000 MHz" [get_ports DDR3B_REFCLK_p]
create_clock -period "125.000000 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



set_false_path -to {clk_sync}

set_false_path -to {sync_chain_halffull[*]}

set_false_path -from {debouncer:e_debouncer|o_q[0]}

set_min_delay -to {xcvr_a10:*|av_ctrl.readdata[*]} -100
set_max_delay -to {xcvr_a10:*|av_ctrl.readdata[*]} 100

set_min_delay -to {readregs[*]} -100
set_max_delay -to {readregs[*]} 100

set_min_delay -to {writeregs_slow[*]} -100
set_max_delay -to {writeregs_slow[*]} 100

set_min_delay -from {writeregs_slow[10][*]} -100
set_max_delay -from {writeregs_slow[10][*]} 100

set_min_delay -from {regwritten[*]} -100
set_max_delay -from {regwritten[*]} 100

# DDR3 paths
#set_false_path -from [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}] -to [get_clocks {ddr3_b|ddr3_A|emif_0_core_usr_clk}]
#set_false_path -from [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}] -to [get_clocks {ddr3_b|ddr3_B|emif_0_core_usr_clk}]
#set_false_path -from [get_clocks {ddr3_b|ddr3_A|emif_0_core_usr_clk}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {ddr3_b|ddr3_B|emif_0_core_usr_clk}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {ddr3_b|ddr3_A|emif_0_phy_clk_l_0}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {ddr3_b|ddr3_B|emif_0_phy_clk_l_0}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {ddr3_b|ddr3_A|emif_0_phy_clk_0}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {ddr3_b|ddr3_B|emif_0_phy_clk_0}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]

#set_false_path -from [get_clocks {rec_switching|xcvr_native_a10_0|g_xcvr_native_insts[0]|rx_clkout}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {rec_switching|xcvr_native_a10_0|g_xcvr_native_insts[1]|rx_clkout}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {rec_switching|xcvr_native_a10_0|g_xcvr_native_insts[2]|rx_clkout}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {rec_switching|xcvr_native_a10_0|g_xcvr_native_insts[3]|rx_clkout}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]

#set_false_path -from [get_clocks {DDR3A_REFCLK_p}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]
#set_false_path -from [get_clocks {DDR3B_REFCLK_p}] -to [get_clocks {pcie_b|pcie_if|pcie_a10_hip_0|coreclkout}]


#Cross clock state markers
#set_false_path -from {data_flow:dataflow|A_writestate}
#set_false_path -from {data_flow:dataflow|B_writestate}
#set_false_path -from {data_flow:dataflow|A_readstate}
#set_false_path -from {data_flow:dataflow|B_readstate}
#set_false_path -from {data_flow:dataflow|A_done}
#set_false_path -from {data_flow:dataflow|B_done}
#
#set_false_path -from {ddr3_block:ddr3_b|ddr3_memory_controller:memctlA|mode.countertest} -to {ddr3_block:ddr3_b|ddr3_if:ddr3_A|ddr3_if_altera_emif_181_57nbgza:emif_0|ddr3_if_altera_emif_arch_nf_181_qf3d2ni:arch|ddr3_if_altera_emif_arch_nf_181_qf3d2ni_top:arch_inst|altera_emif_arch_nf_io_tiles_wrap:io_tiles_wrap_inst|altera_emif_arch_nf_io_tiles:io_tiles_inst|tile_gen[*].lane_gen[*].lane_inst~phy_reg1}
#set_false_path -from {ddr3_block:ddr3_b|ddr3_memory_controller:memctlA|mode.dataflow} -to {ddr3_block:ddr3_b|ddr3_if:ddr3_A|ddr3_if_altera_emif_181_57nbgza:emif_0|ddr3_if_altera_emif_arch_nf_181_qf3d2ni:arch|ddr3_if_altera_emif_arch_nf_181_qf3d2ni_top:arch_inst|altera_emif_arch_nf_io_tiles_wrap:io_tiles_wrap_inst|altera_emif_arch_nf_io_tiles:io_tiles_inst|tile_gen[*].lane_gen[*].lane_inst~phy_reg1}
#set_false_path -from {ddr3_block:ddr3_b|ddr3_memory_controller:memctlB|mode.countertest} -to {ddr3_block:ddr3_b|ddr3_if:ddr3_B|ddr3_if_altera_emif_181_57nbgza:emif_0|ddr3_if_altera_emif_arch_nf_181_qf3d2ni:arch|ddr3_if_altera_emif_arch_nf_181_qf3d2ni_top:arch_inst|altera_emif_arch_nf_io_tiles_wrap:io_tiles_wrap_inst|altera_emif_arch_nf_io_tiles:io_tiles_inst|tile_gen[*].lane_gen[*].lane_inst~phy_reg1}
#set_false_path -from {ddr3_block:ddr3_b|ddr3_memory_controller:memctlB|mode.dataflow} -to {ddr3_block:ddr3_b|ddr3_if:ddr3_B|ddr3_if_altera_emif_181_57nbgza:emif_0|ddr3_if_altera_emif_arch_nf_181_qf3d2ni:arch|ddr3_if_altera_emif_arch_nf_181_qf3d2ni_top:arch_inst|altera_emif_arch_nf_io_tiles_wrap:io_tiles_wrap_inst|altera_emif_arch_nf_io_tiles:io_tiles_inst|tile_gen[*].lane_gen[*].lane_inst~phy_reg1}

