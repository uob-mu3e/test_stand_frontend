~/ghdl/bin/ghdl -a --ieee=synopsys -fexplicit --syn-binding ip_ram.vhd mp8_sc_master.vhd mp8_slowcontrol.vhd mupix_block.vhd spi_if_write_bits.vhd spi_master.vhd mupix_block_tb.vhd dac_fifo.vhd fifo.vhd
~/ghdl/bin/ghdl -e --ieee=synopsys -fexplicit --syn-binding mupix_block_tb
~/ghdl/bin/ghdl -r --ieee=synopsys -fexplicit --syn-binding mupix_block_tb --stop-time=10000ns --vcd=out.vcd
rm *.o *.cf mupix_block_tb
gtkwave out.vcd 
