
CABLE?=1

IPs = \
    nios.tcl

include util/quartus/makefile.mk

.PHONY: maxpof
maxpof : app $(SOF)
	quartus_cpf -c pofconversion.cof

.PHONY: maxpgm
maxpgm : maxpof
	quartus_pgm -m jtag -c $(CABLE) --operation="p;output_files/max10_top.pof"
