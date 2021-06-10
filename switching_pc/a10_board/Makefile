
CABLE=DE5
SOF?=output_files/top.sof

IPs = \
    $(wildcard a10/ip/*.tcl) \
    $(wildcard a10/pcieapp/*.tcl) \
    $(wildcard a10/*.vhd.envsubst) \
    nios.tcl

include util/altera/makefile.mk

clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak
