#

foreach { module_name dir } { temac_gbe_v9_0_rgmii .cache/ } {
    create_ip -vendor xilinx.com -library ip \
              -name tri_mode_ethernet_mac -version 9.0 \
              -module_name $module_name -dir $dir

    foreach { name value } [ list \
        CONFIG.Physical_Interface RGMII \
        CONFIG.MAC_Speed 1000_Mbps \
        CONFIG.Management_Interface false \
    ] {
        set_property $name $value [ get_ips $module_name ]
    }
}
