
CABLE=1

IPs = \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard ip/*.tcl)) \
    a10/ip_xcvr_phy.sopcinfo a10/ip_xcvr_fpll.sopcinfo a10/ip_xcvr_reset.sopcinfo \
    nios.sopcinfo

include util/altera/makefile.mk
