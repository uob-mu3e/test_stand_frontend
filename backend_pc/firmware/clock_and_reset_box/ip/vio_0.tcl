#

foreach { module_name dir } { vio_0 .cache/ } {
    create_ip -vendor xilinx.com -library ip \
              -name vio -version 3.0 \
              -module_name $module_name -dir $dir

    set_property -dict [ list \
        CONFIG.C_EN_PROBE_IN_ACTIVITY 1 \
        CONFIG.C_NUM_PROBE_IN 1 \
    ] [ get_ips $module_name ]
}
