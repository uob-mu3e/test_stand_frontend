
CABLE=DE5

IPs = \
    $(wildcard ip/*.tcl) \
    $(wildcard pcieapp/*.tcl) \
    $(wildcard ddr3/*.tcl) \
    nios.tcl

include util/altera/makefile.mk

clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak *.qsys
