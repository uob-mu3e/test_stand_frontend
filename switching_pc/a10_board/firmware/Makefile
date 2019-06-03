
CABLE=1
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs=\
ip/ip_xcvr_phy.sopcinfo \
ip/ip_xcvr_reset.sopcinfo \
ip/ip_xcvr_fpll.sopcinfo \
ip/ip_clkctrl.sopcinfo \
nios.sopcinfo
VHDs=top.vhd a10/xcvr_a10.vhd sc_master.vhd sc_slave.vhd

include util/makefile.mk

.PRECIOUS: ip/%.qsys
ip/%.qsys : a10/%.tcl
	qsys-script --script=a10/$*.tcl


clean :
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo *.bak
