#

package require qsys

set dir0 [ file dirname [ info script ] ]

source [ file join $dir0 "../device.tcl" ]
source [ file join $dir0 "../util/altera_ip.tcl" ]

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_iopll 50.0 125.0
set_instance_parameter_value iopll_0 {gui_pll_auto_reset} {1}
save_system [ file join $dir0 "$name.qsys" ]
