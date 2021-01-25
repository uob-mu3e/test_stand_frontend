#

source "device.tcl"
source "util/altera_ip.tcl"

add_ram_2port 32 65536 -dc -widthB 128 -regA -regB
