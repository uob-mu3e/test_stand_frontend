onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/i_coreclk
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/i_data
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/i_rst
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/i_SC_disable_dec
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/i_valid
add wave -noupdate -group DUT -radix hexadecimal -childformat {{/testbench/dut/o_data(63) -radix hexadecimal} {/testbench/dut/o_data(62) -radix hexadecimal} {/testbench/dut/o_data(61) -radix hexadecimal} {/testbench/dut/o_data(60) -radix hexadecimal} {/testbench/dut/o_data(59) -radix hexadecimal} {/testbench/dut/o_data(58) -radix hexadecimal} {/testbench/dut/o_data(57) -radix hexadecimal} {/testbench/dut/o_data(56) -radix hexadecimal} {/testbench/dut/o_data(55) -radix hexadecimal} {/testbench/dut/o_data(54) -radix hexadecimal} {/testbench/dut/o_data(53) -radix hexadecimal} {/testbench/dut/o_data(52) -radix hexadecimal} {/testbench/dut/o_data(51) -radix hexadecimal} {/testbench/dut/o_data(50) -radix hexadecimal} {/testbench/dut/o_data(49) -radix hexadecimal} {/testbench/dut/o_data(48) -radix hexadecimal} {/testbench/dut/o_data(47) -radix hexadecimal} {/testbench/dut/o_data(46) -radix hexadecimal} {/testbench/dut/o_data(45) -radix hexadecimal} {/testbench/dut/o_data(44) -radix hexadecimal} {/testbench/dut/o_data(43) -radix hexadecimal} {/testbench/dut/o_data(42) -radix hexadecimal} {/testbench/dut/o_data(41) -radix hexadecimal} {/testbench/dut/o_data(40) -radix hexadecimal} {/testbench/dut/o_data(39) -radix hexadecimal} {/testbench/dut/o_data(38) -radix hexadecimal} {/testbench/dut/o_data(37) -radix hexadecimal} {/testbench/dut/o_data(36) -radix hexadecimal} {/testbench/dut/o_data(35) -radix hexadecimal} {/testbench/dut/o_data(34) -radix hexadecimal} {/testbench/dut/o_data(33) -radix hexadecimal} {/testbench/dut/o_data(32) -radix hexadecimal} {/testbench/dut/o_data(31) -radix hexadecimal} {/testbench/dut/o_data(30) -radix hexadecimal} {/testbench/dut/o_data(29) -radix hexadecimal} {/testbench/dut/o_data(28) -radix hexadecimal} {/testbench/dut/o_data(27) -radix hexadecimal} {/testbench/dut/o_data(26) -radix hexadecimal} {/testbench/dut/o_data(25) -radix hexadecimal} {/testbench/dut/o_data(24) -radix hexadecimal} {/testbench/dut/o_data(23) -radix hexadecimal} {/testbench/dut/o_data(22) -radix hexadecimal} {/testbench/dut/o_data(21) -radix hexadecimal} {/testbench/dut/o_data(20) -radix hexadecimal} {/testbench/dut/o_data(19) -radix hexadecimal} {/testbench/dut/o_data(18) -radix hexadecimal} {/testbench/dut/o_data(17) -radix hexadecimal} {/testbench/dut/o_data(16) -radix hexadecimal} {/testbench/dut/o_data(15) -radix hexadecimal} {/testbench/dut/o_data(14) -radix hexadecimal} {/testbench/dut/o_data(13) -radix hexadecimal} {/testbench/dut/o_data(12) -radix hexadecimal} {/testbench/dut/o_data(11) -radix hexadecimal} {/testbench/dut/o_data(10) -radix hexadecimal} {/testbench/dut/o_data(9) -radix hexadecimal} {/testbench/dut/o_data(8) -radix hexadecimal} {/testbench/dut/o_data(7) -radix hexadecimal} {/testbench/dut/o_data(6) -radix hexadecimal} {/testbench/dut/o_data(5) -radix hexadecimal} {/testbench/dut/o_data(4) -radix hexadecimal} {/testbench/dut/o_data(3) -radix hexadecimal} {/testbench/dut/o_data(2) -radix hexadecimal} {/testbench/dut/o_data(1) -radix hexadecimal} {/testbench/dut/o_data(0) -radix hexadecimal}} -subitemconfig {/testbench/dut/o_data(63) {-height 17 -radix hexadecimal} /testbench/dut/o_data(62) {-height 17 -radix hexadecimal} /testbench/dut/o_data(61) {-height 17 -radix hexadecimal} /testbench/dut/o_data(60) {-height 17 -radix hexadecimal} /testbench/dut/o_data(59) {-height 17 -radix hexadecimal} /testbench/dut/o_data(58) {-height 17 -radix hexadecimal} /testbench/dut/o_data(57) {-height 17 -radix hexadecimal} /testbench/dut/o_data(56) {-height 17 -radix hexadecimal} /testbench/dut/o_data(55) {-height 17 -radix hexadecimal} /testbench/dut/o_data(54) {-height 17 -radix hexadecimal} /testbench/dut/o_data(53) {-height 17 -radix hexadecimal} /testbench/dut/o_data(52) {-height 17 -radix hexadecimal} /testbench/dut/o_data(51) {-height 17 -radix hexadecimal} /testbench/dut/o_data(50) {-height 17 -radix hexadecimal} /testbench/dut/o_data(49) {-height 17 -radix hexadecimal} /testbench/dut/o_data(48) {-height 17 -radix hexadecimal} /testbench/dut/o_data(47) {-height 17 -radix hexadecimal} /testbench/dut/o_data(46) {-height 17 -radix hexadecimal} /testbench/dut/o_data(45) {-height 17 -radix hexadecimal} /testbench/dut/o_data(44) {-height 17 -radix hexadecimal} /testbench/dut/o_data(43) {-height 17 -radix hexadecimal} /testbench/dut/o_data(42) {-height 17 -radix hexadecimal} /testbench/dut/o_data(41) {-height 17 -radix hexadecimal} /testbench/dut/o_data(40) {-height 17 -radix hexadecimal} /testbench/dut/o_data(39) {-height 17 -radix hexadecimal} /testbench/dut/o_data(38) {-height 17 -radix hexadecimal} /testbench/dut/o_data(37) {-height 17 -radix hexadecimal} /testbench/dut/o_data(36) {-height 17 -radix hexadecimal} /testbench/dut/o_data(35) {-height 17 -radix hexadecimal} /testbench/dut/o_data(34) {-height 17 -radix hexadecimal} /testbench/dut/o_data(33) {-height 17 -radix hexadecimal} /testbench/dut/o_data(32) {-height 17 -radix hexadecimal} /testbench/dut/o_data(31) {-height 17 -radix hexadecimal} /testbench/dut/o_data(30) {-height 17 -radix hexadecimal} /testbench/dut/o_data(29) {-height 17 -radix hexadecimal} /testbench/dut/o_data(28) {-height 17 -radix hexadecimal} /testbench/dut/o_data(27) {-height 17 -radix hexadecimal} /testbench/dut/o_data(26) {-height 17 -radix hexadecimal} /testbench/dut/o_data(25) {-height 17 -radix hexadecimal} /testbench/dut/o_data(24) {-height 17 -radix hexadecimal} /testbench/dut/o_data(23) {-height 17 -radix hexadecimal} /testbench/dut/o_data(22) {-height 17 -radix hexadecimal} /testbench/dut/o_data(21) {-height 17 -radix hexadecimal} /testbench/dut/o_data(20) {-height 17 -radix hexadecimal} /testbench/dut/o_data(19) {-height 17 -radix hexadecimal} /testbench/dut/o_data(18) {-height 17 -radix hexadecimal} /testbench/dut/o_data(17) {-height 17 -radix hexadecimal} /testbench/dut/o_data(16) {-height 17 -radix hexadecimal} /testbench/dut/o_data(15) {-height 17 -radix hexadecimal} /testbench/dut/o_data(14) {-height 17 -radix hexadecimal} /testbench/dut/o_data(13) {-height 17 -radix hexadecimal} /testbench/dut/o_data(12) {-height 17 -radix hexadecimal} /testbench/dut/o_data(11) {-height 17 -radix hexadecimal} /testbench/dut/o_data(10) {-height 17 -radix hexadecimal} /testbench/dut/o_data(9) {-height 17 -radix hexadecimal} /testbench/dut/o_data(8) {-height 17 -radix hexadecimal} /testbench/dut/o_data(7) {-height 17 -radix hexadecimal} /testbench/dut/o_data(6) {-height 17 -radix hexadecimal} /testbench/dut/o_data(5) {-height 17 -radix hexadecimal} /testbench/dut/o_data(4) {-height 17 -radix hexadecimal} /testbench/dut/o_data(3) {-height 17 -radix hexadecimal} /testbench/dut/o_data(2) {-height 17 -radix hexadecimal} /testbench/dut/o_data(1) {-height 17 -radix hexadecimal} /testbench/dut/o_data(0) {-height 17 -radix hexadecimal}} /testbench/dut/o_data
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/o_valid
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_addr_a
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_addr_b
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_data_bypass
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_data_decE
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_data_decT
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_init
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_init_dec
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_init_prbs
add wave -noupdate -group DUT -radix hexadecimal /testbench/dut/s_select_bypass
add wave -noupdate -radix hexadecimal /testbench/i_coreclk
add wave -noupdate -radix hexadecimal /testbench/i_data
add wave -noupdate -radix hexadecimal /testbench/i_rst
add wave -noupdate -radix hexadecimal /testbench/i_valid
add wave -noupdate -radix hexadecimal /testbench/o_data
add wave -noupdate -radix hexadecimal /testbench/o_valid
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