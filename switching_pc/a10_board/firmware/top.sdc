#

create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2J]
create_clock -period "125.000000 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]

derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



set_false_path -to {clk_sync}

set_false_path -to {sync_chain_halffull[*]}

set_false_path -from {midas_event_builder:e_midas_event_builder|ip_dcfifo:\buffer_link_fifos:*:e_fifo|dcfifo:dcfifo_component|dcfifo_g1r1:auto_generated|altsyncram_ena1:fifo_ram|*} -to {midas_event_builder:e_midas_event_builder|w_ram_data[*]};
set_false_path -from {midas_event_builder:e_midas_event_builder|ip_dcfifo:\buffer_link_fifos:*:e_fifo|dcfifo:dcfifo_component|dcfifo_g1r1:auto_generated|altsyncram_ena1:fifo_ram|*} -to {midas_event_builder:e_midas_event_builder|w_ram_add[*]};
set_false_path -from {midas_event_builder:e_midas_event_builder|ip_dcfifo:\buffer_link_fifos:*:e_fifo|dcfifo:dcfifo_component|dcfifo_g1r1:auto_generated|altsyncram_ena1:fifo_ram|*} -to {midas_event_builder:e_midas_event_builder|w_ram_add_reg[*]};
set_false_path -from {midas_event_builder:e_midas_event_builder|ip_dcfifo:\buffer_link_fifos:*:e_fifo|dcfifo:dcfifo_component|dcfifo_g1r1:auto_generated|altsyncram_ena1:fifo_ram|*} -to {midas_event_builder:e_midas_event_builder|event_tagging_state.*};

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
