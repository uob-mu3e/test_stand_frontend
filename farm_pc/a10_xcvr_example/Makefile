
CABLE=1
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

include util/makefile.mk

flow : ip/ip_xcvr_phy.sopcinfo ip/ip_xcvr_fpll.sopcinfo ip/ip_xcvr_reset.sopcinfo
flow : ip/ip_clkctrl.sopcinfo ip/ip_iopll_125.sopcinfo
flow : nios.sopcinfo

ip/%.qsys : a10/%.tcl
	qsys-script --script=$<
