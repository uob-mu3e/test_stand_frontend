ghdl -a --ieee=synopsys -fexplicit link_tester.vhd link_observer.vhd link_test_tb.vhd
ghdl -e --ieee=synopsys -fexplicit link_test_tb
ghdl -r --ieee=synopsys -fexplicit link_test_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf link_test_tb
gtkwave out.vcd 
