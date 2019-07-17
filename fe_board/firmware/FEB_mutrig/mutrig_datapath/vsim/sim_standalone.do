vsim -novopt  work.testbench_standalone

add wave -noupdate -group DUT -radix hexadecimal /testbench_standalone/dut/*
add wave -noupdate -expand -group MUX -radix hexadecimal /testbench_standalone/dut/u_mux/*

