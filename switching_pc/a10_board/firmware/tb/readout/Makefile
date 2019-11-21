GHDL=ghdl
GHDLFLAGS=
MODULES=\
    data_generator_a10_tb.o \
    ip_ram.o \
    fifo.o \
    readout_tb.o \
    ip_ram \
    fifo \
    data_generator_a10_tb \
    readout_tb


test: $(MODULES)
    ./readout_tb --vcd=readout_tb.vcd

# Binary depends on the object file
%: %.o
    $(GHDL) -e $(GHDLFLAGS) $@

# Object file depends on source
%.o: %.vhd
    $(GHDL) -a $(GHDLFLAGS) $<

clean:
    echo "Cleaning up..."
    rm -f *.o *_testbench ip_ram carry_ripple_adder work*.cf e*.lst