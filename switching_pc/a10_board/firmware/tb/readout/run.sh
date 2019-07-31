ghdl -a --ieee=synopsys -fexplicit linear_shift.vhd event_counter_tb.vhd data_generator_a10_tb.vhd ip_ram.vhd fifo.vhd fifo_36bit.vhd readout_tb.vhd
ghdl -e --ieee=synopsys -fexplicit readout_tb
ghdl -r --ieee=synopsys -fexplicit readout_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf readout_tb
gtkwave out.vcd 
