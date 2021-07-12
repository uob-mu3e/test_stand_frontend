onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -radix hexadecimal /sequencer_tb/reset_n
add wave -noupdate -radix hexadecimal /sequencer_tb/reset
add wave -noupdate -radix hexadecimal /sequencer_tb/writeclk
add wave -noupdate -radix hexadecimal /sequencer_tb/tofifo_counters
add wave -noupdate -radix hexadecimal /sequencer_tb/fromfifo_counters
add wave -noupdate -radix hexadecimal /sequencer_tb/read_counterfifo
add wave -noupdate -radix hexadecimal /sequencer_tb/write_counterfifo
add wave -noupdate -radix hexadecimal /sequencer_tb/counterfifo_almostfull
add wave -noupdate -radix hexadecimal /sequencer_tb/counterfifo_empty
add wave -noupdate -radix hexadecimal /sequencer_tb/outcommand
add wave -noupdate -radix hexadecimal /sequencer_tb/command_enable
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/running
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/running_last
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/output
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/current_block
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/counters_reg
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/subaddr
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/dohits
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/from_fifo_reg
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/read_fifo_int
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/fifo_reg_valid
add wave -noupdate -radix hexadecimal /sequencer_tb/dut/fifo_reg_new
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {0 ps} 0}
quietly wave cursor active 0
configure wave -namecolwidth 505
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
WaveRestoreZoom {13973838 ps} {14136460 ps}
