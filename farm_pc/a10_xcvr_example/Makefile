
CABLE=1
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

include util/makefile.mk

flow : nios.sopcinfo
flow : a10/ip_clkctrl.sopcinfo a10/ip_iopll_125.sopcinfo
flow : a10/ip_xcvr_phy.sopcinfo a10/ip_xcvr_fpll.sopcinfo a10/ip_xcvr_reset.sopcinfo
