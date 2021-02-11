onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /mupix_ctrl_tb/clk
add wave -noupdate /mupix_ctrl_tb/reset_n
add wave -noupdate /mupix_ctrl_tb/reg_re
add wave -noupdate /mupix_ctrl_tb/reg_we
add wave -noupdate /mupix_ctrl_tb/reg_add
add wave -noupdate /mupix_ctrl_tb/reg_rdata
add wave -noupdate /mupix_ctrl_tb/reg_wdata
add wave -noupdate /mupix_ctrl_tb/counter_int
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/slow_down
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/is_writing
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/is_writing_this_round
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/config_storage_write
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/config_storage_input_data
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/enable_shift_reg6
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/clk_step
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/rd_config
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/config_data
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/config_data29
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/wait_cnt
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/ld_regs
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/mp_ctrl_state
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/mp_fifo_clear
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/i_clr_all
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/i_data
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/i_wrreq
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/o_data
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/i_enable
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/i_rdreq
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/fifo_read
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/bitpos
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/bitpos_global
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/is_writing
add wave -noupdate /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_config_storage/enable_prev
add wave -noupdate /mupix_ctrl_tb/clock
add wave -noupdate /mupix_ctrl_tb/mosi
add wave -noupdate /mupix_ctrl_tb/csn

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
force -freeze mupix_ctrl_tb/reg_add x"48" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000" -cancel 8ns
run 16ns
force -freeze mupix_ctrl_tb/reg_add x"41" -cancel 80ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 64ns
force -freeze mupix_ctrl_tb/reg_wdata x"D1AFB54D"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"AB75183F"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"12345678"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"FF10BACE"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"CAFECAFE"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"FEB0FEB1"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"ABCDEF12"
run 80ns
force -freeze mupix_ctrl_tb/reg_add x"40" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000001" -cancel 8ns
run 40ns
force -freeze mupix_ctrl_tb/reg_add x"40" -cancel 8ns
force -freeze mupix_ctrl_tb/reg_we 1 -cancel 8ns
force -freeze mupix_ctrl_tb/reg_wdata x"00000000" -cancel 8ns
run 10ms
WaveRestoreZoom 0ns 1000000ns
