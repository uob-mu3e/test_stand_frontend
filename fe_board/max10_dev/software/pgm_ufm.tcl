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



processor_stop [ lindex $::proc_paths 0 ]
processor_reset [ lindex $::proc_paths 0 ]
mm_claim 0



proc write_srec { mm fname } {
    set f [ open $fname r ]

    while { true } {
        set line [ gets $f ]
        if { [ eof $f ] } {
            break
        }

        set type [ string range $line 0 1 ]

        # header record
        if { $type == "S0" } {
            continue
        }

        # count record
        if { $type == "S5" } {
            continue
        }

        # termination records
        if { $type == "S7" || $type == "S8" || $type == "S9" } {
            continue
        }

        set n [ string range $line 2 3 ]
        set n [ expr 0x$n ]

        # data records
        if { $type == "S1" } {
            set addr [ string range $line 4 7 ]
            set n [ expr $n - 2 ]
        }
        if { $type == "S2" } {
            set addr [ string range $line 4 9 ]
            set n [ expr $n - 3 ]
        }
        if { $type == "S3" } {
            set addr [ string range $line 4 11 ]
            set n [ expr $n - 4 ]
        }
        set addr [ expr 0x$addr ]

        if { $n < 1 } {
            error "error: "
        }

        set bytes [ list ]
        while { $n > 1 } {
            set p [ expr [ string length $line ] - 2 * $n ]
            set b [ string range $line $p [ expr $p + 1 ] ]
            set n [ expr $n - 1 ]
            lappend bytes $b
        }

#        ::ufm::erase 1
#        ::ufm::erase 2
        ::ufm::disable_wp $mm 1
        ::ufm::disable_wp $mm 2
        foreach { a b c d } $bytes {
            set u32 0x$d$c$b$a
            puts [ format "debug: \[0x%08X\] <= 0x%08X" [ expr $addr ] [ expr $u32 ] ]
            master_write_32 $mm $addr $u32
            ::ufm::wait_idle $mm
            if { [ master_read_32 $mm $addr 1 ] != $u32 } {
                puts [ format "warn: \[0x%08X\] = 0x%08X != 0x%08X" [ expr $addr ] [ expr [ master_read_32 $mm $addr 1 ] ] [ expr $u32 ] ]
            }
            set addr [ expr $addr + 4 ]
        }
        ::ufm::enable_wp $mm 1
        ::ufm::enable_wp $mm 2

        # checksum
#        set cs [ string range $line ... ]
    }

    close $f
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
