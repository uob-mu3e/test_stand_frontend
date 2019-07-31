onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -radix hexadecimal /data_flow_tb/reset_n
add wave -noupdate -radix hexadecimal /data_flow_tb/dataclk
add wave -noupdate -radix hexadecimal /data_flow_tb/data_en
add wave -noupdate -radix hexadecimal /data_flow_tb/data_in
add wave -noupdate -radix hexadecimal /data_flow_tb/ts_in
add wave -noupdate -radix hexadecimal /data_flow_tb/pcieclk
add wave -noupdate -radix hexadecimal /data_flow_tb/ts_req_A
add wave -noupdate -radix hexadecimal /data_flow_tb/req_en_A
add wave -noupdate -radix hexadecimal /data_flow_tb/ts_req_B
add wave -noupdate -radix hexadecimal /data_flow_tb/req_en_B
add wave -noupdate -radix hexadecimal /data_flow_tb/tsblock_done
add wave -noupdate -radix hexadecimal /data_flow_tb/dma_data_out
add wave -noupdate -radix hexadecimal /data_flow_tb/dma_data_en
add wave -noupdate -radix hexadecimal /data_flow_tb/dma_eoe
add wave -noupdate -radix hexadecimal /data_flow_tb/link_data_out
add wave -noupdate -radix hexadecimal /data_flow_tb/link_ts_out
add wave -noupdate -radix hexadecimal /data_flow_tb/link_data_en
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_clk
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_ready
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_calibrated
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_addr
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_data
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_write
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_read
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_q
add wave -noupdate -radix hexadecimal /data_flow_tb/A_mem_q_valid
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_clk
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_ready
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_calibrated
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_addr
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_data
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_write
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_read
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_q
add wave -noupdate -radix hexadecimal /data_flow_tb/B_mem_q_valid
add wave -noupdate -radix hexadecimal /data_flow_tb/toggle
add wave -noupdate -radix hexadecimal /data_flow_tb/startinput
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/mem_mode_A
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/mem_mode_B
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/ddr3if_state_A
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/ddr3if_state_B
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tsrange
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/B_tsrange
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/tsupper_last
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_memready
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/B_memready
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_writestate
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/B_writestate
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_readstate
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/B_readstate
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/tofifo_A
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/tofifo_B
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/writefifo_A
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/writefifo_B
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_fifo_empty
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/B_fifo_empty
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_reqfifo_empty
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/B_reqfifo_empty
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagram_write
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagram_data
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagram_q
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagram_address
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagram_addrnext
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagram_datanext
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_mem_addr_reg
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/readfifo_A
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/qfifo_A
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_tagts_last
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_wstarted
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_numwords
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_readreqfifo
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_reqfifoq
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_req_last
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_readsubstate
add wave -noupdate -radix hexadecimal /data_flow_tb/dut/A_readwords
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {0 ps} 0}
quietly wave cursor active 0
configure wave -namecolwidth 150
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
configure wave -timelineunits ns
update
WaveRestoreZoom {4999050 ps} {5000050 ps}
