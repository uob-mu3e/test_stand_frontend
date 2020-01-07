#

create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2J]
create_clock -period "125.000000 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



set_false_path -to {clk_sync}

set_false_path -to {sync_chain_halffull[*]}

set_false_path -from {reset_logic:e_reset_logic|resets_reg[13]} -to {midas_event_builder:e_midas_event_builder|midas_bank_builder:\buffer_banks:*:e_bank|ip_dcfifo:e_fifo|dcfifo:dcfifo_component|dcfifo_g1r1:auto_generated|dffpipe_3dc:rdaclr|dffe13a[0]}; 
set_false_path -from {reset_logic:e_reset_logic|resets_reg[13]} -to {midas_event_builder:e_midas_event_builder|midas_bank_builder:\buffer_banks:*:e_bank|ip_dcfifo:e_fifo|dcfifo:dcfifo_component|dcfifo_g1r1:auto_generated|dffpipe_3dc:rdaclr|dffe12a[0]}; 

#multicycle path for huge combinatorial adder in bank builder
#used in state event_size: +3 use +3
set_multicycle_path -to {midas_event_builder:e_midas_event_builder|all_bank_size[*]} -setup -end 3
#used in state event_size_trailer, bank_size_trailer: far later, use +6
set_multicycle_path -to {midas_event_builder:e_midas_event_builder|event_size_int[*]} -setup -end 6
#used in state event_bank_size: +4 use +4
set_multicycle_path -to {midas_event_builder:e_midas_event_builder|event_data_size[*]} -setup -end 4

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
