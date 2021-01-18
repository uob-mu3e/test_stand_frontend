#

package require qsys

source "device.tcl"
source "util/altera_ip.tcl"

set name [ file tail [ file rootname [ info script ] ] ]

create_system $name
add_altera_iopll 125.0 312.5 -locked
