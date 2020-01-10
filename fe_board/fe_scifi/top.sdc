#

#false path for slow control registers (156MHz) to receiver clock (125M)
#signals are quasi-static and have synchronizers
set_false_path -through {scifi_path:e_scifi_path|s_subdet_reset_reg[*]}
set_false_path -through {scifi_path:e_scifi_path|s_dummyctrl_reg[*]}
set_false_path -through {scifi_path:e_scifi_path|s_dpctrl_reg[*]}

#clock domain crossing 125MHz lvds-pll clock  --> core clock on FIFO edge. FIFOused is gray encoded and synced, should be fine
set_false_path -to {scifi_path:e_scifi_path|mutrig_datapath:e_mutrig_datapath|mutrig_store:\g_buffer:*:u_elastic_buffer|s_fifoused_reg[*]}

#reset alignment block: fals paths between slow and fast shifted clock, user needs to find proper dynamic phase shift and cycle settings
set_false_path -to [get_clocks {e_scifi_path|u_resetshift|e_pll|altpll_component|auto_generated|pll1|clk[0]}]
set_false_path -to [get_clocks {e_scifi_path|u_resetshift|e_pll|altpll_component|auto_generated|pll1|clk[1]}]

#status flags to registers
set_false_path -from {scifi_path:e_scifi_path|mutrig_datapath:e_mutrig_datapath|receiver_block:u_rxdeser|data_decoder:\gen_channels:*:datadec|ready_buf} -to {scifi_path:e_scifi_path|o_reg_rdata[*]}
#set_false_path -through {e_scifi_path|e_mutrig_datapath|u_rxdeser|lvds_rx|ALTLVDS_RX_component|auto_generated|rx_dpa_locked[*]|combout} -to {scifi_path:e_scifi_path|o_reg_rdata[*]}
set_false_path -to {scifi_path:e_scifi_path|rx_dpa_lock_reg[*]}

#counters (gray encoded)
set_false_path -to {scifi_path:e_scifi_path|s_cntreg_num[*]}
set_false_path -to {scifi_path:e_scifi_path|s_cntreg_denom_g_156[*]}
#counter reset.
#set_false_path -from {scifi_path:e_scifi_path|s_cntreg_ctrl[15]}
