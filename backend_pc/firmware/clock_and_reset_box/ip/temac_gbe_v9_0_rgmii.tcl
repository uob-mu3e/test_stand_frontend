#

set module_name temac_gbe_v9_0_rgmii
set dir .cache/

create_ip -vlnv xilinx.com:ip:tri_mode_ethernet_mac:9.0 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    CONFIG.Physical_Interface RGMII \
    CONFIG.MAC_Speed 1000_Mbps \
    CONFIG.Management_Interface false \
] [ get_ips $module_name ]

