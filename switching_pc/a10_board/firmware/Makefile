
CABLE=DE5
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs= \
    a10/ip_xcvr_phy.sopcinfo \
    a10/ip_xcvr_reset.sopcinfo \
    a10/ip_xcvr_fpll.sopcinfo \
    a10/ip_clkctrl.sopcinfo \
    nios.sopcinfo
VHDs= \
    a10/xcvr_a10.vhd \
    sc_master.vhd \
    sc_slave.vhd \
    top.vhd

include util/makefile.mk

clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak
