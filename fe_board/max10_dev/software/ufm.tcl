#
# author : Alexandr Kozlinskiy
# date : 2019-08-26
#

namespace eval ::ufm {
    set CSR 0x700F00F0
    set STATUS [ expr $CSR + 4 * 0 ]
    set CONTROL [ expr $CSR + 4 * 1 ]
}

proc ::ufm::wait_idle { mm { ms 10 } } {
    while { [ master_read_32 $mm $::ufm::STATUS 1 ] & 0x3 } {
        puts [ master_read_32 $mm $::ufm::STATUS 1 ]
        after $ms
    }
}

proc ::ufm::disable_wp { mm sector } {
    set sector [ expr $sector + 22 ]
    set control [ master_read_32 $mm $::ufm::CONTROL 1 ]
    master_write_32 $mm $::ufm::CONTROL [ expr $control & ~(1 << $sector) ]
}

proc ::ufm::enable_wp { mm sector } {
    set sector [ expr $sector + 22 ]
    set control [ master_read_32 $mm $::ufm::CONTROL 1 ]
    master_write_32 $mm $::ufm::CONTROL [ expr $control | (1 << $sector) ]
}

proc ::ufm::ws { mm } {
    set control [ master_read_32 $mm $::ufm::CONTROL 1 ]
    return [ expr ($control & 0x08) == 0x08 ]
}
