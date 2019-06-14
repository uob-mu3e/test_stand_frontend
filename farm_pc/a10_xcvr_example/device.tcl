set_project_property DEVICE_FAMILY {Arria 10}
set_project_property DEVICE {10AX115N2F45I1SG}

set nios_freq 125000000

set refclk_freq 125000000
set txrx_data_rate 5000

#if { $txrx_data_rate / 40 != $refclk_freq * 1e-6 } {
#    error "mismatch between 'txrx_data_rate' and 'refclk_freq'"
#}
