onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /testbench/i_A_data
add wave -noupdate /testbench/i_A_valid
add wave -noupdate /testbench/i_coreclk
add wave -noupdate /testbench/i_rst
add wave -noupdate /testbench/o_A_data
add wave -noupdate /testbench/o_A_valid
add wave -noupdate /testbench/o_initializing
add wave -noupdate -expand -group DUT /testbench/dut/i_A_data
add wave -noupdate -expand -group DUT /testbench/dut/i_A_valid
add wave -noupdate -expand -group DUT /testbench/dut/i_coreclk
add wave -noupdate -expand -group DUT /testbench/dut/i_rst
add wave -noupdate -expand -group DUT /testbench/dut/i_SC_disable_dec
add wave -noupdate -expand -group DUT /testbench/dut/n_addr_a
add wave -noupdate -expand -group DUT /testbench/dut/o_A_data
add wave -noupdate -expand -group DUT /testbench/dut/o_A_valid
add wave -noupdate -expand -group DUT /testbench/dut/o_initializing
add wave -noupdate -expand -group DUT /testbench/dut/s_addr_a
add wave -noupdate -expand -group DUT /testbench/dut/s_data_bypass_0
add wave -noupdate -expand -group DUT /testbench/dut/s_data_bypass_1
add wave -noupdate -expand -group DUT /testbench/dut/s_data_bypass_2
add wave -noupdate -expand -group DUT /testbench/dut/s_data_dec
add wave -noupdate -expand -group DUT /testbench/dut/s_init
add wave -noupdate -expand -group DUT /testbench/dut/s_init_dec
add wave -noupdate -expand -group DUT /testbench/dut/s_init_dec_d
add wave -noupdate -expand -group DUT /testbench/dut/s_init_prbs
add wave -noupdate -expand -group DUT /testbench/dut/s_is_header_0
add wave -noupdate -expand -group DUT /testbench/dut/s_is_header_1
add wave -noupdate -expand -group DUT /testbench/dut/s_is_header_2
add wave -noupdate -expand -group DUT /testbench/dut/s_is_header_3
add wave -noupdate -expand -group DUT /testbench/dut/s_select_bypass_0
add wave -noupdate -expand -group DUT /testbench/dut/s_select_bypass_1
add wave -noupdate -expand -group DUT /testbench/dut/s_select_bypass_2
add wave -noupdate -expand -group DUT /testbench/dut/s_valid_1
add wave -noupdate -expand -group DUT /testbench/dut/s_valid_2
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {262189468 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 150
configure wave -valuecolwidth 146
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
WaveRestoreZoom {262020606 ps} {263402208 ps}
