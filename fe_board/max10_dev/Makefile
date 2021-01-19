
.ONESHELL:

CABLE=1

IPs = \
    adc_lab_proj/myadc.tcl \
    nios.tcl

include util/altera/makefile.mk

top.qsf : top.qip
	cat << EOF > $@
	set_global_assignment -name QIP_FILE $<
	set_global_assignment -name TOP_LEVEL_ENTITY top
	set_global_assignment -name PRE_FLOW_SCRIPT_FILE "quartus_sh:util/altera/pre_flow.tcl"
	set_global_assignment -name PROJECT_OUTPUT_DIRECTORY output_files
	EOF

all : top.qsf
