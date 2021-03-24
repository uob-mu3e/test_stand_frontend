onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /mupix_ctrl_dummy_tb/clk
add wave -noupdate /mupix_ctrl_dummy_tb/reset_n
add wave -noupdate /mupix_ctrl_dummy_tb/clock
add wave -noupdate /mupix_ctrl_dummy_tb/mosi
add wave -noupdate /mupix_ctrl_dummy_tb/csn
add wave -noupdate /mupix_ctrl_dummy_tb/e_mupix_ctrl/step
add wave -noupdate /mupix_ctrl_dummy_tb/e_mupix_ctrl/e_mupix_ctrl/e_mupix_ctrl_config_storage/bitpos_global
add wave -noupdate /mupix_ctrl_dummy_tb/e_mupix_ctrl/e_mupix_ctrl/e_mupix_ctrl_config_storage/o_data


TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {4127596 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 367
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
radix -hexadecimal

run 5ms

WaveRestoreZoom 0ns 1000000ns
