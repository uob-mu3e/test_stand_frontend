
CABLE=DE5
SOF=output_files/top.sof

BSP_DIR=software/hal_bsp
APP_DIR=software/app

VHDs= top.vhd \
      mudaq_components.vhd \
      mudaq_registers.vhd \
      pcieapp/*.vhd \
      helpers/*.vhd \
      ddr3/*.vhd

version_reg.vhd: FORCE
	tclsh revision_number.tcl

FORCE: ;

clean:
	rm -rf *.rpt *.chg smart.log *.htm *.eqn *.pin *.sof *.pof db incremental_db *.sopcinfo pcieapp/completer_fifo pcieapp/completer_wide_fifo pcieapp/data_addrs_ram pcieapp/data_pages_ram pcieapp/dma_ram pcieapp/pcie pcieapp/pcie_ram_narrow pcieapp/pcie_wram_narrow simulation helpers/ip_clk_ctrl helpers/receiver_switching *.bak helpers/*.sopcinfo helpers/ip_receiving_fifo

include util/makefile.mk


