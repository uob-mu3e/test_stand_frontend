
CABLE?=1

IPs = \
    nios.tcl

include util/altera/makefile.mk

.PHONY: maxpof
maxpof : app $(SOF)
	quartus_cpf -c pofconversion.cof

.PHONY: maxpgm
maxpgm : maxpof
	quartus_pgm -m jtag -c $(CABLE) --operation="p;output_files/max10_top.pof"

top.qsf : top.qip
	cat << EOF > $@
	set_global_assignment -name QIP_FILE $<
	set_global_assignment -name TOP_LEVEL_ENTITY top
	set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
	set_global_assignment -name QIP_FILE $(PREFIX)/include.qip
	EOF

all : top.qsf
