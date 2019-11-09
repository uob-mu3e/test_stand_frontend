#

foreach { module_name dir } { ila_0 .cache/ } {
    create_ip -vendor xilinx.com -library ip \
              -name ila -version 6.2 \
              -module_name $module_name -dir $dir

    foreach { name value } [ list \
        CONFIG.C_NUM_OF_PROBES 4 \
        CONFIG.C_DATA_DEPTH 131072 \
        CONFIG.C_PROBE0_WIDTH 1 \
        CONFIG.C_PROBE1_WIDTH 1 \
        CONFIG.C_PROBE2_WIDTH 1 \
        CONFIG.C_PROBE3_WIDTH 32 \
    ] {
        set_property $name $value [ get_ips $module_name ]
    }
}
