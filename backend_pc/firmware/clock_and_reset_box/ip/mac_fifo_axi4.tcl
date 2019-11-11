#

set module_name mac_fifo_axi4
set dir .cache/

create_ip -vlnv xilinx.com:ip:fifo_generator:13.2 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    {CONFIG.Clock_Type_AXI} {Independent_Clock} \
    {CONFIG.Empty_Threshold_Assert_Value_axis} {13} \
    {CONFIG.Empty_Threshold_Assert_Value_rach} {13} \
    {CONFIG.Empty_Threshold_Assert_Value_rdch} {1021} \
    {CONFIG.Empty_Threshold_Assert_Value_wach} {13} \
    {CONFIG.Empty_Threshold_Assert_Value_wdch} {1021} \
    {CONFIG.Empty_Threshold_Assert_Value_wrch} {13} \
    {CONFIG.Enable_Safety_Circuit} {true} \
    {CONFIG.Enable_TLAST} {true} \
    {CONFIG.FIFO_Implementation_axis} {Independent_Clocks_Distributed_RAM} \
    {CONFIG.FIFO_Implementation_rach} {Independent_Clocks_Distributed_RAM} \
    {CONFIG.FIFO_Implementation_rdch} {Independent_Clocks_Block_RAM} \
    {CONFIG.FIFO_Implementation_wach} {Independent_Clocks_Distributed_RAM} \
    {CONFIG.FIFO_Implementation_wdch} {Independent_Clocks_Block_RAM} \
    {CONFIG.FIFO_Implementation_wrch} {Independent_Clocks_Distributed_RAM} \
    {CONFIG.Full_Flags_Reset_Value} {1} \
    {CONFIG.Full_Threshold_Assert_Value_axis} {15} \
    {CONFIG.Full_Threshold_Assert_Value_rach} {15} \
    {CONFIG.Full_Threshold_Assert_Value_wach} {15} \
    {CONFIG.Full_Threshold_Assert_Value_wrch} {15} \
    {CONFIG.INTERFACE_TYPE} {AXI_STREAM} \
    {CONFIG.Input_Depth_axis} {16} \
    {CONFIG.Reset_Type} {Asynchronous_Reset} \
    {CONFIG.TUSER_WIDTH} {1} \
] [ get_ips $module_name ]
