#
# author : Alexandr Kozlinskiy
# date : 2019-08-26
#

source [ file join [ file dirname [ info script ] ] "mm.tcl" ]
source [ file join [ file dirname [ info script ] ] "ufm.tcl" ]
source [ file join [ file dirname [ info script ] ] "srec.tcl" ]



mm_claim 0



proc write_srec { mm fname } {
    set f [ open $fname r ]

    while { true } {
        set record [ gets $f ]
        if { [ eof $f ] } {
            break
        }

        set addr 0
        set bytes [ srec::parse_record $record addr ]

        foreach { a b c d } $bytes {
            set u32 0x$d$c$b$a
            if { [ master_read_32 $mm $addr 1 ] != $u32 } {
                puts [ format "debug: \[0x%08X\] <= 0x%08X" [ expr $addr ] [ expr $u32 ] ]
                if { $addr < 0x8000 } {
                    ::ufm::write $mm $addr $u32
                } \
                else {
                    master_write_32 $mm $addr $u32
                }
            }
            set addr [ expr $addr + 4 ]
        }
    }

    close $f
}

proc pgm { mm } {
    set proc_paths [ get_service_paths processor ]
    processor_stop [ lindex $proc_paths 0 ]

#    ::ufm::erase $mm 1
#    ::ufm::erase $mm 2
    write_srec $mm "software/app/main.srec"

    processor_reset [ lindex $proc_paths 0 ]
    processor_run [ lindex $proc_paths 0 ]
}

proc test_read { mm } {
    for { set i 0 } { $i < 16 } { incr i } {
        set addr [ expr 0x00000000 + 4 * $i ]
        set data [ master_read_32 $mm $addr 1 ]
        puts [ format "0x%08X" $data ]
    }
}

proc test_write { mm } {
    for { set i 0 } { $i < 32 } { incr i } {
        set addr [ expr 0x00000000 + 4 * $i ]
        ::ufm::write $mm $addr $i
    }
}
