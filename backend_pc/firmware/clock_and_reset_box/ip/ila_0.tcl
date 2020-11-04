#

set module_name ila_0
set dir .cache/

create_ip -vlnv xilinx.com:ip:ila:6.2 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    {CONFIG.C_DATA_DEPTH} {4096} \
    {CONFIG.C_NUM_OF_PROBES} {16} \
    {CONFIG.C_PROBE2_WIDTH} {32} \
    {CONFIG.C_PROBE5_WIDTH} {32} \
    {CONFIG.C_PROBE8_WIDTH} {32} \
    {CONFIG.C_PROBE11_WIDTH} {32} \
] [ get_ips $module_name ]
