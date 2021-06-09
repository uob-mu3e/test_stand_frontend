vlib work
vmap work work

vcom -work work ../altera/ip_ram.vhd
vcom -work work ../histogram_generic_half_rate.vhd
vcom -work work ../histogram_generic.vhd
vcom -work work tb_histogram_generic.vhd

set TOP_LEVEL_NAME histogram_generic_vhd_tst

vsim -L altera_mf histogram_generic_vhd_tst
do wave_histogram_generic.do
run 20 us