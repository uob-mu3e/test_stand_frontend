ghdl -a --ieee=synopsys -fexplicit dma_counter.vhd halffull_tb.vhd
ghdl -e --ieee=synopsys -fexplicit halffull_tb
ghdl -r --ieee=synopsys -fexplicit halffull_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf halffull_tb
gtkwave out.vcd 
