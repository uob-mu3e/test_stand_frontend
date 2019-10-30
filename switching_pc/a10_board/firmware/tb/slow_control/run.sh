ghdl -a --ieee=synopsys -fexplicit sc_master.vhd sc_slave.vhd sram.vhd sc_tb.vhd
ghdl -e --ieee=synopsys -fexplicit sc_tb
ghdl -r --ieee=synopsys -fexplicit sc_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf sc_tb
gtkwave out.vcd 
