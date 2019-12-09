#

set module_name temac_gbe_v9_0_rgmii
set dir .cache/

create_ip -vlnv xilinx.com:ip:tri_mode_ethernet_mac:9.0 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    {CONFIG.Enable_MDIO} {false} \
    {CONFIG.MAC_Speed} {1000_Mbps} \
    {CONFIG.Make_MDIO_External} {false} \
    {CONFIG.Management_Interface} {false} \
    {CONFIG.Number_of_Table_Entries} {0} \
    {CONFIG.Physical_Interface} {RGMII} \
    {CONFIG.Statistics_Counters} {false} \
] [ get_ips $module_name ]

