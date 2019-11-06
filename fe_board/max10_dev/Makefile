
CABLE=1
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

IPs = \
    nios.sopcinfo adc_lab_proj/myadc.sopcinfo
VHDs = \
    top.vhd

adc_lab_proj/myadc.sopcinfo : adc_lab_proj/myadc.tcl
	( cd adc_lab_proj && qsys-script --script=myadc.tcl && qsys-generate --synthesis=VHDL myadc.qsys )

include util/makefile.mk

nios.qsys : device.tcl
