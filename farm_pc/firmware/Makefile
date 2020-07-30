
CABLE=DE5
SOF?=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs = \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard ip/*.tcl)) \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard pcieapp/*.tcl)) \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard ddr3/*.tcl)) \
    a10/ip_xcvr_phy.sopcinfo a10/ip_xcvr_reset.sopcinfo a10/ip_xcvr_fpll.sopcinfo \
    nios.sopcinfo

include util/makefile.mk

clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak
