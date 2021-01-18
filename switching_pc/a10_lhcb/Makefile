
CABLE=1

IPs := \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard ip/*.tcl)) \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard pcieapp/*.tcl)) \
    nios.sopcinfo

include util/altera/makefile.mk
