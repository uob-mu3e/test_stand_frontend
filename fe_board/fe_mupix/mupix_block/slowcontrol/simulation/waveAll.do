quit -sim
vlib work
project compileall
vsim work.mupix_ctrl_tb(rtl)

onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /mupix_ctrl_tb/clk
add wave -noupdate /mupix_ctrl_tb/reset_n
add wave -noupdate /mupix_ctrl_tb/reg_we
add wave -noupdate /mupix_ctrl_tb/reg_add
add wave -noupdate /mupix_ctrl_tb/reg_wdata
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/o_data
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/i_rdreq
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/o_is_writing
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/fifo_write_final
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/fifo_wdata_final

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
force -freeze mupix_ctrl_tb/reg_re 0
run 80ns
force -freeze mupix_ctrl_tb/reg_add x"47" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000003" -cancel 8ns
run 16ns
force -freeze mupix_ctrl_tb/reg_add x"49" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000002" -cancel 8ns
run 16ns
force -freeze mupix_ctrl_tb/reg_add x"48" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000" -cancel 8ns
run 16ns
force -freeze mupix_ctrl_tb/reg_add x"4A" -cancel 672ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 672ns
force -freeze mupix_ctrl_tb/reg_wdata x"2A000A03"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"FA3F002F"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"1E041041"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"041E9A51"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"40280000"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"1400C20A"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"028A001F"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00020038"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"0000FC09"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"F0001C80"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00148000"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"11802E00"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000"
run 800ns


run 4ms

WaveRestoreZoom 0ns 1000000ns
