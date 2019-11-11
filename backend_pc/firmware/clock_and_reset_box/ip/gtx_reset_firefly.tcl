#

set module_name gtx_reset_firefly
set dir .cache/

create_ip -vlnv xilinx.com:ip:gtwizard:3.6 \
          -module_name $module_name -dir $dir

set_property -dict [ list \
    CONFIG.gt0_usesharedlogic 1 \
    CONFIG.identical_protocol_file aurora_8b10b_multi_lane_4byte \
    CONFIG.identical_val_no_rx true \
    CONFIG.identical_val_tx_line_rate 1.25 \
    CONFIG.identical_val_tx_reference_clock 125.000 \
    CONFIG.gt0_val false \
    CONFIG.gt1_val false \
    CONFIG.gt2_val true \
    CONFIG.gt2_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt2_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt3_val true \
    CONFIG.gt3_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt3_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt4_val true \
    CONFIG.gt4_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt4_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt5_val true \
    CONFIG.gt5_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt5_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt6_val true \
    CONFIG.gt6_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt6_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt7_val true \
    CONFIG.gt7_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt7_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt8_val true \
    CONFIG.gt8_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt8_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt9_val true \
    CONFIG.gt9_val_rx_refclk REFCLK0_Q1 \
    CONFIG.gt9_val_tx_refclk REFCLK0_Q1 \
    CONFIG.gt0_uselabtools true \
    CONFIG.gt0_val_drp_clock 31.25 \
    CONFIG.gt0_val_txbuf_en false \
] [ get_ips $module_name ]
