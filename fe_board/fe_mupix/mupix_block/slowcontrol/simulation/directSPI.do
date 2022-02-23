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
add wave -noupdate -group mp_ctrl /mupix_ctrl_tb/e_mp_ctrl/*
add wave -noupdate -group mp_ctrl_regs /mupix_ctrl_tb/e_mp_ctrl/e_mupix_ctrl_reg_mapping/*
add wave -noupdate -group direct_spi /mupix_ctrl_tb/e_mp_ctrl/gen_spi(0)/mp_ctrl_direct_spi_inst/*
add wave -noupdate -group direct_spi_fifo /mupix_ctrl_tb/e_mp_ctrl/gen_spi(0)/mp_ctrl_direct_spi_inst/direct_spi_fifo/*

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
force -freeze mupix_ctrl_tb/reg_add [examine mupix_registers/MP_CTRL_SLOW_DOWN_REGISTER_W]
force -freeze mupix_ctrl_tb/reg_wdata "x00000004"
force -freeze mupix_ctrl_tb/reg_we 1
run 8ns
force -freeze mupix_ctrl_tb/reg_add [examine mupix_registers/MP_CTRL_DIRECT_SPI_ENABLE_REGISTER_W]
force -freeze mupix_ctrl_tb/reg_wdata "x00000001"
run 8ns 
force -freeze mupix_ctrl_tb/reg_add [examine mupix_registers/MP_CTRL_DIRECT_SPI_START_REGISTER_W]
force -freeze mupix_ctrl_tb/reg_wdata "xCCAACCAA"
run 8ns
force -freeze mupix_ctrl_tb/reg_we 0
run 80000 ns

WaveRestoreZoom 0ns 10000ns
