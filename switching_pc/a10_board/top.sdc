#

create_clock -period  "50.001 MHz" [get_ports CLK_50_B2J]
create_clock -period "125.002 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.003 MHz" [get_ports PCIE_REFCLK_p]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty

# false path from rx.locked bits to pcie registers
set_false_path -to {a10_block:a10_block|local_pcie0_rregs_A[53][*]}
set_false_path -to {a10_block:a10_block|local_pcie0_rregs_A[54][*]}