# qsys scripting (.tcl) file for dflowfifo
package require -exact qsys 16.0

create_system {dflowfifo}

set_project_property DEVICE_FAMILY {Arria 10}
set_project_property DEVICE {10AX115N2F45I1SG}
set_project_property HIDE_FROM_IP_CATALOG {true}

# Instances and instance parameters
# (disabled instances are intentionally culled)
add_instance fifo_0 fifo 18.1
set_instance_parameter_value fifo_0 {GUI_AlmostEmpty} {0}
set_instance_parameter_value fifo_0 {GUI_AlmostEmptyThr} {1}
set_instance_parameter_value fifo_0 {GUI_AlmostFull} {0}
set_instance_parameter_value fifo_0 {GUI_AlmostFullThr} {1}
set_instance_parameter_value fifo_0 {GUI_CLOCKS_ARE_SYNCHRONIZED} {0}
set_instance_parameter_value fifo_0 {GUI_Clock} {4}
set_instance_parameter_value fifo_0 {GUI_DISABLE_DCFIFO_EMBEDDED_TIMING_CONSTRAINT} {1}
set_instance_parameter_value fifo_0 {GUI_Depth} {256}
set_instance_parameter_value fifo_0 {GUI_ENABLE_ECC} {0}
set_instance_parameter_value fifo_0 {GUI_Empty} {1}
set_instance_parameter_value fifo_0 {GUI_Full} {1}
set_instance_parameter_value fifo_0 {GUI_LE_BasedFIFO} {0}
set_instance_parameter_value fifo_0 {GUI_LegacyRREQ} {0}
set_instance_parameter_value fifo_0 {GUI_MAX_DEPTH} {Auto}
set_instance_parameter_value fifo_0 {GUI_MAX_DEPTH_BY_9} {0}
set_instance_parameter_value fifo_0 {GUI_OVERFLOW_CHECKING} {0}
set_instance_parameter_value fifo_0 {GUI_Optimize} {0}
set_instance_parameter_value fifo_0 {GUI_Optimize_max} {0}
set_instance_parameter_value fifo_0 {GUI_RAM_BLOCK_TYPE} {Auto}
set_instance_parameter_value fifo_0 {GUI_UNDERFLOW_CHECKING} {0}
set_instance_parameter_value fifo_0 {GUI_UsedW} {1}
set_instance_parameter_value fifo_0 {GUI_Width} {272}
set_instance_parameter_value fifo_0 {GUI_dc_aclr} {1}
set_instance_parameter_value fifo_0 {GUI_delaypipe} {5}
set_instance_parameter_value fifo_0 {GUI_diff_widths} {0}
set_instance_parameter_value fifo_0 {GUI_msb_usedw} {0}
set_instance_parameter_value fifo_0 {GUI_output_width} {8}
set_instance_parameter_value fifo_0 {GUI_read_aclr_synch} {0}
set_instance_parameter_value fifo_0 {GUI_rsEmpty} {1}
set_instance_parameter_value fifo_0 {GUI_rsFull} {0}
set_instance_parameter_value fifo_0 {GUI_rsUsedW} {0}
set_instance_parameter_value fifo_0 {GUI_sc_aclr} {0}
set_instance_parameter_value fifo_0 {GUI_sc_sclr} {0}
set_instance_parameter_value fifo_0 {GUI_synStage} {3}
set_instance_parameter_value fifo_0 {GUI_write_aclr_synch} {0}
set_instance_parameter_value fifo_0 {GUI_wsEmpty} {0}
set_instance_parameter_value fifo_0 {GUI_wsFull} {1}
set_instance_parameter_value fifo_0 {GUI_wsUsedW} {1}

# exported interfaces
set_instance_property fifo_0 AUTO_EXPORT {true}

# interconnect requirements
set_interconnect_requirement {$system} {qsys_mm.clockCrossingAdapter} {HANDSHAKE}
set_interconnect_requirement {$system} {qsys_mm.enableEccProtection} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.insertDefaultSlave} {FALSE}
set_interconnect_requirement {$system} {qsys_mm.maxAdditionalLatency} {1}

save_system {dflowfifo.qsys}
