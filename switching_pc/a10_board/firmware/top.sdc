#

create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2J]
create_clock -period "125.000000 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



set_false_path -to {clk_sync}

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