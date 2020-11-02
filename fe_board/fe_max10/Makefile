
CABLE?=1
SOF?=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs = \
    nios.sopcinfo

include util/makefile.mk

nios.qsys : device.tcl

.PHONY: maxpof
maxpof : app $(SOF)
	quartus_cpf -c pofconversion.cof

.PHONY: maxpgm
maxpgm : maxpof
	quartus_pgm -m jtag -c $(CABLE) --operation="p;output_files/max10_top.pof"
