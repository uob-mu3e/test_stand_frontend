onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /feb_common_tb/clk
add wave -noupdate /feb_common_tb/reset
add wave -noupdate /feb_common_tb/state_idle
add wave -noupdate /feb_common_tb/state_run_prepare
add wave -noupdate /feb_common_tb/state_sync
add wave -noupdate /feb_common_tb/state_running
add wave -noupdate /feb_common_tb/state_terminating
add wave -noupdate /feb_common_tb/state_link_test
add wave -noupdate /feb_common_tb/state_sync_test
add wave -noupdate /feb_common_tb/state_reset
add wave -noupdate /feb_common_tb/state_out_of_DAQ
add wave -noupdate /feb_common_tb/data_out
add wave -noupdate /feb_common_tb/data_is_k
add wave -noupdate /feb_common_tb/data_in
add wave -noupdate /feb_common_tb/data_in_slowcontrol
add wave -noupdate /feb_common_tb/slowcontrol_fifo_empty
add wave -noupdate /feb_common_tb/data_fifo_empty
add wave -noupdate /feb_common_tb/slowcontrol_read_req
add wave -noupdate /feb_common_tb/data_read_req
add wave -noupdate /feb_common_tb/terminated
add wave -noupdate /feb_common_tb/reset_link
add wave -noupdate /feb_common_tb/runnumber
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {225425 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 332
configure wave -valuecolwidth 277
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
WaveRestoreZoom {0 ps} {289673 ps}
run
force -freeze sim:/feb_common_tb/reset_link x\"bc\" 0
force -freeze sim:/feb_common_tb/data_fifo_empty 1 0
force -freeze sim:/feb_common_tb/slowcontrol_fifo_empty 1 0
run
run
run
force -freeze sim:/feb_common_tb/reset_link x\"10\" 0
run
force -freeze sim:/feb_common_tb/reset_link x\"05\" 0
run
run
force -freeze sim:/feb_common_tb/reset_link x\"bc\" 0
run
run
run
run
run
run
force -freeze sim:/feb_common_tb/reset_link x\"11\" 0
run
run
run
force -freeze sim:/feb_common_tb/reset_link x\"12\" 0
run
run
run
run
force -freeze sim:/feb_common_tb/data_in 001001111111110101010101010101010101 0
force -freeze sim:/feb_common_tb/data_fifo_empty 0 0
run
force -freeze sim:/feb_common_tb/data_in 000001111111110101010101010101010101 0
run
run
run
force -freeze sim:/feb_common_tb/data_in_slowcontrol 001001010101010101010101010101010101 0
force -freeze sim:/feb_common_tb/slowcontrol_fifo_empty 0 0
run
run
run
force -freeze sim:/feb_common_tb/data_in 001101111111110101010101010101010101 0
run
run
run
force -freeze sim:/feb_common_tb/data_in_slowcontrol 000001010101010101010101010101010101 0
run
run
run
run
run
run
force -freeze sim:/feb_common_tb/data_in_slowcontrol 001101010101010101010101010101010101 0
run
force -freeze sim:/feb_common_tb/slowcontrol_fifo_empty 1 0
run
run
force -freeze sim:/feb_common_tb/data_in 000001111111110101010101010101010101 0
run
run 
run 
run
run
force -freeze sim:/feb_common_tb/reset_link x\"13\" 0
run
run
run
run
force -freeze sim:/feb_common_tb/data_in 001101111111110101010101010101010101 0
run
run
run
run
run
run
run
run
run
run
run
