#

source software/include/hal_bsp.tcl

# sections
delete_memory_region flash_data
add_memory_region flash flash_data 0x0000 0x8000
