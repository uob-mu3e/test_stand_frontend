onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /tb_sc_new/clk
add wave -noupdate /tb_sc_new/reset_n
add wave -noupdate /tb_sc_new/data_in
add wave -noupdate /tb_sc_new/datak_in
add wave -noupdate /tb_sc_new/data_out
add wave -noupdate /tb_sc_new/data_out_we
add wave -noupdate /tb_sc_new/sc_reg
add wave -noupdate /tb_sc_new/fe_reg
add wave -noupdate /tb_sc_new/sc_ram
add wave -noupdate /tb_sc_new/subdet_reg
add wave -noupdate /tb_sc_new/av_sc_address
add wave -noupdate /tb_sc_new/av_sc_read
add wave -noupdate /tb_sc_new/av_sc_readdata
add wave -noupdate /tb_sc_new/av_sc_write
add wave -noupdate /tb_sc_new/av_sc_writedata
add wave -noupdate /tb_sc_new/av_sc_waitrequest



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

run 1ms

WaveRestoreZoom 0ns 1000000ns
