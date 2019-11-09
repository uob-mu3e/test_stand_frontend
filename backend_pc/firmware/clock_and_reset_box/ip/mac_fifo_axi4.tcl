#

foreach { module_name dir } { mac_fifo_axi4 .cache/ } {
    create_ip -vendor xilinx.com -library ip \
              -name fifo_generator -version 13.2 \
              -module_name $module_name -dir $dir

    foreach { name value } [ list \
        CONFIG.INTERFACE_TYPE AXI_STREAM \
        CONFIG.Clock_Type_AXI Independent_Clock \
        CONFIG.TUSER_WIDTH 1 \
        CONFIG.Enable_TLAST true \
        CONFIG.FIFO_Implementation_axis Independent_Clocks_Distributed_RAM \
        CONFIG.Input_Depth_axis 16 \
    ] {
        set_property $name $value [ get_ips $module_name ]
    }
}
