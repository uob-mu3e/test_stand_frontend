ghdl -a --ieee=synopsys -fexplicit counter.vhd dma_test_tb.vhd
ghdl -e --ieee=synopsys -fexplicit dma_test_tb
ghdl -r --ieee=synopsys -fexplicit dma_test_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf dma_test_tb
gtkwave out.vcd 
