#

source "device.tcl"
source "util/altera_ip.tcl"

add_altera_iopll 50.0 125.0
set_instance_parameter_value iopll_0 {gui_pll_auto_reset} {1}
