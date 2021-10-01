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
force -freeze mupix_ctrl_tb/reg_add x"41" -cancel 80ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 64ns
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
force -freeze mupix_ctrl_tb/reg_wdata x"028A0000"
run 80ns


force -freeze mupix_ctrl_tb/reg_add x"42" -cancel 80ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 64ns
force -freeze mupix_ctrl_tb/reg_wdata x"001F0002"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00380000"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"FC09F000"
run 80ns


force -freeze mupix_ctrl_tb/reg_add x"43" -cancel 80ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 32ns
force -freeze mupix_ctrl_tb/reg_wdata x"00720000"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"52000046"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00B80000"
run 80ns

force -freeze mupix_ctrl_tb/reg_add x"44" -cancel 240ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 240ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000"
run 248ns

force -freeze mupix_ctrl_tb/reg_add x"45" -cancel 240ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 240ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000"
run 248ns

force -freeze mupix_ctrl_tb/reg_add x"46" -cancel 240ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 240ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000"
run 248ns

run 80ns

force -freeze mupix_ctrl_tb/reg_add x"40" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"0000003F" -cancel 8ns
run 40ns
force -freeze mupix_ctrl_tb/reg_add x"40" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000" -cancel 8ns
run 4ms

WaveRestoreZoom 0ns 1000000ns
