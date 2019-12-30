onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group dut /testbench_standalone/dut/GEN_DUMMIES
add wave -noupdate -group dut /testbench_standalone/dut/INPUT_SIGNFLIP
add wave -noupdate -group dut /testbench_standalone/dut/LVDS_DATA_RATE
add wave -noupdate -group dut /testbench_standalone/dut/LVDS_PLL_FREQ
add wave -noupdate -group dut /testbench_standalone/dut/N_ASICS
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_counterselect
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_datagen_count
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_datagen_enable
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_datagen_shortmode
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_disable_dec
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_mask
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_reset_counters
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_rx_wait_for_all
add wave -noupdate -group dut /testbench_standalone/dut/i_SC_rx_wait_for_all_sticky
add wave -noupdate -group dut /testbench_standalone/dut/i_clk_core
add wave -noupdate -group dut /testbench_standalone/dut/i_fifo_rd
add wave -noupdate -group dut /testbench_standalone/dut/i_refclk_125_A
add wave -noupdate -group dut /testbench_standalone/dut/i_refclk_125_B
add wave -noupdate -group dut /testbench_standalone/dut/i_rst
add wave -noupdate -group dut /testbench_standalone/dut/i_stic_txd
add wave -noupdate -group dut /testbench_standalone/dut/i_ts_clk
add wave -noupdate -group dut /testbench_standalone/dut/i_ts_rst
add wave -noupdate -group dut /testbench_standalone/dut/o_buffer_full
add wave -noupdate -group dut /testbench_standalone/dut/o_counter_denominator_high
add wave -noupdate -group dut /testbench_standalone/dut/o_counter_denominator_low
add wave -noupdate -group dut /testbench_standalone/dut/o_counter_nominator
add wave -noupdate -group dut /testbench_standalone/dut/o_fifo_data
add wave -noupdate -group dut /testbench_standalone/dut/o_fifo_empty
add wave -noupdate -group dut /testbench_standalone/dut/o_frame_desync
add wave -noupdate -group dut /testbench_standalone/dut/o_receivers_dpa_lock
add wave -noupdate -group dut /testbench_standalone/dut/o_receivers_pll_lock
add wave -noupdate -group dut /testbench_standalone/dut/o_receivers_ready
add wave -noupdate -group dut /testbench_standalone/dut/o_receivers_usrclk
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_almost_full
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_data
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_full
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_predec_data
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_predec_full
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_predec_wr
add wave -noupdate -group dut /testbench_standalone/dut/s_buf_wr
add wave -noupdate -group dut /testbench_standalone/dut/s_crc_error
add wave -noupdate -group dut /testbench_standalone/dut/s_crcerrorcounter
add wave -noupdate -group dut /testbench_standalone/dut/s_end_of_frame
add wave -noupdate -group dut /testbench_standalone/dut/s_event_data
add wave -noupdate -group dut /testbench_standalone/dut/s_event_ready
add wave -noupdate -group dut /testbench_standalone/dut/s_eventcounter
add wave -noupdate -group dut /testbench_standalone/dut/s_fifos_data
add wave -noupdate -group dut /testbench_standalone/dut/s_fifos_empty
add wave -noupdate -group dut /testbench_standalone/dut/s_fifos_full
add wave -noupdate -group dut /testbench_standalone/dut/s_fifos_rd
add wave -noupdate -group dut /testbench_standalone/dut/s_frame_info
add wave -noupdate -group dut /testbench_standalone/dut/s_frame_info_rdy
add wave -noupdate -group dut /testbench_standalone/dut/s_frame_number
add wave -noupdate -group dut /testbench_standalone/dut/s_framecounter
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_busy
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_end_of_frame
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_event_data
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_event_ready
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_frame_info
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_frame_info_rdy
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_frame_number
add wave -noupdate -group dut /testbench_standalone/dut/s_gen_new_frame
add wave -noupdate -group dut /testbench_standalone/dut/s_new_frame
add wave -noupdate -group dut /testbench_standalone/dut/s_prbs_err_cnt
add wave -noupdate -group dut /testbench_standalone/dut/s_prbs_wrd_cnt
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_end_of_frame
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_event_data
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_event_ready
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_frame_info
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_frame_info_rdy
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_frame_number
add wave -noupdate -group dut /testbench_standalone/dut/s_rec_new_frame
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_all_ready
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_block
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_data
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_data_isk
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_errorcounter
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_ready
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_runcounter
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_state
add wave -noupdate -group dut /testbench_standalone/dut/s_receivers_usrclk
add wave -noupdate -group dut /testbench_standalone/dut/s_timecounter
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/N_INPUTID_BITS
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/N_INPUTS
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_SC_mask
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_SC_nomerge
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_coreclk
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_rst
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_sink_full
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_source_data
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_source_empty
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_timestamp_clk
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/i_timestamp_rst
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_all_header
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_all_trailer
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_asic_hitdropped
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_asic_overflow
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_crc_err
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_common_data
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_frameid_nonsync
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_request
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/l_request_next
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/n_Hpart
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/n_Tpart
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/n_is_valid
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/n_sel_gnt
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/n_sink_wr
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/n_state
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/o_sink_data
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/o_sink_wr
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/o_source_rd
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/o_sync_error
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_Hpart
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_Tpart
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_chnum
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_global_timestamp
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_is_valid
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_read
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_sel_data
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_sel_gnt
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_sink_wr
add wave -noupdate -group Mux -radix hexadecimal /testbench_standalone/dut/u_mux/s_state
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {0 ps} 0}
quietly wave cursor active 0
configure wave -namecolwidth 452
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {0 ps} {690 ps}
