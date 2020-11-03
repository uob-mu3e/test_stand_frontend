
CABLE=1
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs := \
    $(patsubst %.tcl,%.sopcinfo,$(wildcard ../fe/ip/*.tcl)) \
    nios.sopcinfo

include util/makefile.mk
