
CABLE=1

IPs := \
    $(wildcard a10/ip/*.tcl) \
    $(wildcard a10/pcieapp/*.tcl) \
    nios.tcl

include util/altera/makefile.mk

output_files/top.pof : output_files/top.sof
	quartus_cpf -c top.cof

pgm_pof : output_files/top.pof
	quartus_pgm -c $(CABLE) top.cdf
