
CABLE?=2

IPs := \
    ../fe/ip/clk_ctrl_single.tcl \
    $(wildcard a5/*.vhd.qmegawiz) \
    nios.tcl

include util/altera/makefile.mk

output_files/top.rbf : output_files/top.sof
	quartus_cpf -c $< $@
