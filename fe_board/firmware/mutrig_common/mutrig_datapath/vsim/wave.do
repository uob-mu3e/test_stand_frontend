onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group TB /testbench/a_sipm_in_n
add wave -noupdate -group TB /testbench/a_sipm_in_p
add wave -noupdate -group TB /testbench/asic_clk
add wave -noupdate -group TB /testbench/asic_clk_common
add wave -noupdate -group TB /testbench/asic_cs
add wave -noupdate -group TB /testbench/asics_miso
add wave -noupdate -group TB /testbench/asics_mosi
add wave -noupdate -group TB /testbench/asics_rst
add wave -noupdate -group TB /testbench/asics_sclk
add wave -noupdate -group TB /testbench/i_asic_tx_p
add wave -noupdate -group TB /testbench/i_coreclk
add wave -noupdate -group TB /testbench/i_refclk_125
add wave -noupdate -group TB /testbench/i_rst
add wave -noupdate -group TB /testbench/o_receivers_dpa_lock
add wave -noupdate -group TB /testbench/o_receivers_pll_clock
add wave -noupdate -group TB /testbench/o_receivers_pll_lock
add wave -noupdate -group TB /testbench/o_receivers_ready
add wave -noupdate -group TB /testbench/s_fifo_data
add wave -noupdate -group TB /testbench/s_fifo_empty
add wave -noupdate -group TB /testbench/s_fifo_rd
add wave -noupdate -group TB /testbench/s_fifo_rd_last
add wave -noupdate -group TB /testbench/s_fsync_err
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_clk_core
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_fifo_rd
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_refclk_125
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_rst
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_SC_disable_dec
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_SC_mask
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/i_stic_txd
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/N_ASICS
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/o_fifo_data
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/o_fifo_empty
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/o_receivers_dpa_lock
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/o_receivers_pll_lock
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/o_receivers_ready
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/o_receivers_usrclk
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_almost_full
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_data
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_full
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_predec_data
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_predec_full
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_predec_wr
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_buf_wr
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_crc_error
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_end_of_frame
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_event_data
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_event_ready
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_eventcounter
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_fifos_data
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_fifos_empty
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_fifos_full
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_fifos_rd
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_frame_info
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_frame_info_rdy
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_frame_number
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_fsync_err
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_new_frame
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_receivers_data
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_receivers_data_isk
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_receivers_ready
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_receivers_state
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_receivers_usrclk
add wave -noupdate -expand -group DUT -radix hexadecimal /testbench/dut/s_timecounter
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {20440316 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 205
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
WaveRestoreZoom {0 ps} {44789220 ps}
