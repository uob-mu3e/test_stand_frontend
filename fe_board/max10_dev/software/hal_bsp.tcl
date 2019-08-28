#

source "software/include/hal_bsp.tcl"

# sections
foreach { name slave offset span } [ get_memory_region flash_data ] {
    set span [ expr 0x8000 - $offset ]
    delete_memory_region flash_data
    add_memory_region flash $slave $offset $span
}
