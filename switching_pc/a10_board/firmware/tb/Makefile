
GHDL=ghdl
WORK=work
GHDLFLAGS=--ieee=synopsys -fexplicit
MODULES=\
    sc_master \
    sc_slave  \
    sram \
    sc_tb

test: group compile
	ghdl -r --ieee=synopsys -fexplicit sc_tb --vcd=sc.vcd --stop-time=400ns

# group all files of the project
group:
	$(GHDL) -i --workdir=$(WORK) *.vhd

# Object file depends on source
compile:
	$(GHDL) -m $(GHDLFLAGS) --workdir=$(WORK) sc_tb

clean:
	echo "Cleaning up..."
	rm -f *.o *_tb sc_master sram work*.cf e*.lst $(WORK)/*
