#

set module_name ila_0
set dir .cache/

create_ip -vlnv xilinx.com:ip:ila:6.2 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    CONFIG.C_NUM_OF_PROBES 4 \
    CONFIG.C_DATA_DEPTH 131072 \
    CONFIG.C_PROBE0_WIDTH 1 \
    CONFIG.C_PROBE1_WIDTH 1 \
    CONFIG.C_PROBE2_WIDTH 1 \
    CONFIG.C_PROBE3_WIDTH 32 \
] [ get_ips $module_name ]
