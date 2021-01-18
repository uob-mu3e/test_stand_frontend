#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_iopll 50.0 125.0
set_instance_parameter_value iopll_0 {gui_pll_auto_reset} {1}
