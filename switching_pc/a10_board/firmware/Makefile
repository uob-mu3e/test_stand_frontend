
CABLE=DE5

IPs = \
    $(wildcard ip/*.tcl) \
    $(wildcard pcieapp/*.tcl) \
    a10/ip_xcvr_phy.tcl a10/ip_xcvr_reset.tcl a10/ip_xcvr_fpll.tcl \
    nios.tcl

include util/altera/makefile.mk

clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak
