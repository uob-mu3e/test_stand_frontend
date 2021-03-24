#

source "device.tcl"
source "util/altera_ip.tcl"

add_ram_2port 64 4096 -2rw -rdw old -regA -regB
