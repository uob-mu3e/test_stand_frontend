
CABLE=1
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs = \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard ip/*.tcl)) \
    a10/ip_xcvr_phy.sopcinfo a10/ip_xcvr_fpll.sopcinfo a10/ip_xcvr_reset.sopcinfo \
    nios.sopcinfo

include util/makefile.mk
