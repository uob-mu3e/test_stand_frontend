
CABLE=DE5
SOF?=output_files/top.sof

IPs = \
    $(wildcard ip/*.tcl) \
    $(wildcard pcieapp/*.tcl) \
    nios.tcl

include util/altera/makefile.mk

clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak
