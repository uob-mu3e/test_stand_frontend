#
# author : Alexandr Kozlinskiy
# date : 2019-08-26
#

source [ file join [ file dirname [ info script ] ] "ufm.tcl" ]



set proc_paths [ get_service_paths processor ]
set mm_paths [ get_service_paths master ]
set mm_index -1

proc mm_claim { { index -1 } } {
    if { $index == $::mm_index } {
        return
    }
    if { [ info exists ::mm ] } {
        ::close_service master $::mm
        unset ::mm
    }
    if { $index >= 0 } {
        set path [ lindex $::mm_paths $index ]
        puts "INFO: claim master '$path'"
        set ::mm [ ::claim_service master $path "" ]
    }
    set ::mm_index $index
}

mm_claim 1



proc read_srec { f } {
    set words [ list ]

    for { set l 0 } { $l < 64 } { incr l } {
        set type [ read $f 2 ]
        if { $type != "S3" } {
            error "error: unknown record type $type"
        }
        set n [ read $f 2 ]
        set n [ expr 0x$n - 5 ]
        if { $n != 32 } {
            error "error: n != 32"
        }
        set addr [ read $f 8 ]

        for { set i 0 } { $i < 32 } { incr i } {
            set b [ read $f 2 ]
            if { [ eof $f ] } break
            lappend words "0x$b"
        }

        set c [ read $f 2 ]
        if { [ read $f 1 ] != "\n" } {
            error "error: expect line feed"
        }
    }

    return $words
}

proc test_read { mm } {
    for { set i 0 } { $i < 16 } { incr i } {
        set addr [ expr 0x00000000 + 4 * $i ]
        set data [ master_read_32 $mm $addr 1 ]
        puts [ format "0x%08X" $data ]
    }
}

proc test_write { mm } {
    ::ufm::disable_wp $mm 1
    for { set i 0 } { $i < 16 } { incr i } {
        set addr [ expr 0x00000000 + 4 * $i ]
        master_write_32 $mm $addr $i
        ::ufm::wait_idle $mm
        puts [ ::ufm::ws $mm ]
    }
    ::ufm::enable_wp $mm 1
}
