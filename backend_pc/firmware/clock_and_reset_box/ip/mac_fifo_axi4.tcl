#

set module_name mac_fifo_axi4
set dir .cache/

create_ip -vlnv xilinx.com:ip:fifo_generator:13.2 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    CONFIG.INTERFACE_TYPE AXI_STREAM \
    CONFIG.Clock_Type_AXI Independent_Clock \
    CONFIG.TUSER_WIDTH 1 \
    CONFIG.Enable_TLAST true \
    CONFIG.FIFO_Implementation_axis Independent_Clocks_Distributed_RAM \
    CONFIG.Input_Depth_axis 16 \
] [ get_ips $module_name ]
