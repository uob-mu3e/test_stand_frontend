onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -group tb -radix hexadecimal /readout_tb/*
add wave -noupdate -expand -group dut -radix hexadecimal /readout_tb/e_midas_event_builder/*
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {446112 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 219
configure wave -valuecolwidth 86
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
configure wave -timelineunits ns
update
WaveRestoreZoom {0 ps} {1638274 ps}
