#

set module_name vio_0
set dir .cache/

create_ip -vlnv xilinx.com:ip:vio:3.0 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    CONFIG.C_EN_PROBE_IN_ACTIVITY {0} \
    CONFIG.C_NUM_PROBE_IN {0} \
] [ get_ips $module_name ]
