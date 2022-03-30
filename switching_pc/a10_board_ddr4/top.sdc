#

create_clock -period  "50.001 MHz" [get_ports CLK_50_B2J]
create_clock -period "125.002 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.003 MHz" [get_ports PCIE_REFCLK_p]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



set_false_path -to {clk_sync}

set_false_path -to {sync_chain_halffull[*]}

set_false_path -from {debouncer:e_debouncer|o_q[0]}

set_min_delay -to {readregs[*]} -100
set_max_delay -to {readregs[*]} 100
