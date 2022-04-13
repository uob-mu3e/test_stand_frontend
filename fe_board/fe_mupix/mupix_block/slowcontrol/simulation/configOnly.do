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
add wave -noupdate -group spi /mupix_ctrl_tb/e_mp_ctrl/gen_spi(0)/mp_ctrl_spi_inst/*
add wave -noupdate -group direct_spi /mupix_ctrl_tb/e_mp_ctrl/gen_spi(0)/mp_ctrl_direct_spi_inst/*
add wave -noupdate -group direct_spi_fifo /mupix_ctrl_tb/e_mp_ctrl/gen_spi(0)/mp_ctrl_direct_spi_inst/direct_spi_fifo/*
add wave -noupdate -group conf_storage /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/*
add wave -noupdate -group conf_splitter /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/mp_conf_splitter/*
add wave -noupdate -group bias_dpf /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/gen_dp_fifos(0)/bias/*
add wave -noupdate -group conf_dpf /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/gen_dp_fifos(0)/conf/*
add wave -noupdate -group vdac_dpf /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/gen_dp_fifos(0)/vdac/*
add wave -noupdate -group tdac_dpf /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/gen_dp_fifos(0)/tdac/*
add wave -noupdate -group tdac_mem /mupix_ctrl_tb/e_mp_ctrl/mupix_ctrl_config_storage_inst/tdac_memory/*

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
force -freeze mupix_ctrl_tb/reg_wdata "x00000000"
run 8ns
force -freeze mupix_ctrl_tb/reg_add [examine mupix_registers/MP_CTRL_COMBINED_START_REGISTER_W]
force -freeze mupix_ctrl_tb/reg_wdata "x2A000A03"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "xFA3F002F"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x1E041041"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x041E9A51"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x40280000"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x1400C20A"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x0280001F"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x00020038"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x0000FC09"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "xF0001C80"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x00148000"
run 8ns
force -freeze mupix_ctrl_tb/reg_wdata "x11802E00"
run 8ns
force -freeze mupix_ctrl_tb/reg_we 0

-- everything:
run 4000000 ns

-- begin only:
--run 160000 ns

WaveRestoreZoom 0ns 1000000ns