ghdl -a --ieee=synopsys -fexplicit ip_ram.vhd mp8_sc_master.vhd mp8_slowcontrol.vhd mupix_block.vhd spi_if_write_bits.vhd spi_master.vhd mupix_block_tb.vhd
ghdl -e --ieee=synopsys -fexplicit mupix_block_tb
ghdl -r --ieee=synopsys -fexplicit mupix_block_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf mupix_block_tb
gtkwave out.vcd 
