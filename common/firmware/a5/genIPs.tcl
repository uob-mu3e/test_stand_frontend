qexec "qmegawiz -silent a5/ip_alt_xcvr_reconfig.vhd"
qexec "qmegawiz -silent a5/ip_altera_xcvr_native_av.vhd"
qexec "qmegawiz -silent a5/ip_altera_xcvr_reset_control.vhd" 

post_message "Transceiver IP generation done"
