onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_rst
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_stic_txd
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_refclk_125_A
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_refclk_125_B
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_ts_clk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_ts_rst
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_clk_core
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_fifo_empty
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_fifo_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_fifo_rd
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_disable_dec
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_mask
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_datagen_enable
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_datagen_shortmode
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_datagen_count
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_rx_wait_for_all
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_usrclk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_pll_lock
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_dpa_lock
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_frame_desync
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_buffer_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_state
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_data_isk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_usrclk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_all_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_crc_error
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_frame_number
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_frame_number
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_frame_number
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_frame_info
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_frame_info
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_frame_info
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_new_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_new_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_new_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_frame_info_rdy
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_frame_info_rdy
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_frame_info_rdy
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_event_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_event_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_event_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_event_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_event_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_event_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_end_of_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_end_of_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_end_of_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_empty
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_rd
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_predec_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_predec_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_predec_wr
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_almost_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_wr
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_eventcounter
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_timecounter
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_rst
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_stic_txd
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_refclk_125_A
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_refclk_125_B
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_ts_clk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_ts_rst
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_clk_core
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_fifo_empty
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_fifo_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_fifo_rd
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_disable_dec
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_mask
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_datagen_enable
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_datagen_shortmode
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_datagen_count
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_rx_wait_for_all
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/i_SC_rx_wait_for_all_sticky
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_usrclk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_pll_lock
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_dpa_lock
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_receivers_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_frame_desync
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/o_buffer_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_state
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_data_isk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_usrclk
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_all_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_receivers_block
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_crc_error
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_frame_number
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_frame_number
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_frame_number
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_frame_info
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_frame_info
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_frame_info
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_new_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_new_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_new_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_frame_info_rdy
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_frame_info_rdy
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_frame_info_rdy
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_event_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_event_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_event_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_event_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_event_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_event_ready
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_end_of_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_rec_end_of_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_gen_end_of_frame
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_empty
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_rd
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_predec_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_predec_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_predec_wr
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_data
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_almost_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_buf_wr
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_fifos_full
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_eventcounter
add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/s_timecounter
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_coreclk
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_rst
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_timestamp_clk
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_timestamp_rst
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_source_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_source_empty
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_source_rd
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_sink_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_sink_full
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_sink_wr
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_sync_error
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_SC_mask
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_SC_nomerge
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_all_header
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_all_trailer
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_frameid_nonsync
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_crc_err
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_asic_overflow
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_asic_hitdropped
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_common_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_global_timestamp
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_is_valid
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_is_valid
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_request
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_read
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_sel_gnt
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_sel_gnt
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_sel_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_chnum
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_Tpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_Tpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_Hpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_Hpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_state
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_state
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_sink_wr
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_sink_wr
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_coreclk
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_rst
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_timestamp_clk
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_timestamp_rst
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_source_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_source_empty
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_source_rd
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_sink_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_sink_full
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_sink_wr
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/o_sync_error
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_SC_mask
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/i_SC_nomerge
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_all_header
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_all_trailer
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_frameid_nonsync
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_crc_err
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_asic_overflow
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_any_asic_hitdropped
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_common_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_global_timestamp
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_is_valid
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_is_valid
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/l_request
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_read
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_sel_gnt
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_sel_gnt
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_sel_data
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_chnum
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_Tpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_Tpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_Hpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_Hpart
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_state
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_state
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/s_sink_wr
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/n_sink_wr
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/i_coreclk
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/i_data
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/i_rst
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/i_SC_disable_dec
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/l_is_header
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/n_addr_a
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/n_is_header
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/n_select_bypass
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/o_data
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/o_valid
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_addr_a
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_data_bypass
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_data_bypass_1
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_data_bypass_2
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_data_dec
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_init
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_init_dec
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_init_dec_d
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_init_prbs
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_is_header
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_select_bypass
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/i_valid
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_valid_2
add wave -noupdate -expand -group decoder /testbench_standalone/dut/u_decoder/s_valid_1
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {162438 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 194
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
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
WaveRestoreZoom {0 ps} {1050 ns}
