set_project_property DEVICE_FAMILY {Arria 10}
set_project_property DEVICE {10AX115N2F45E1SG}

set refclk_freq 50.0
set txrx_data_rate 2000

if { $txrx_data_rate / 40 != $refclk_freq } {
    error "mismatch between 'txrx_data_rate' and 'refclk_freq'"
}
