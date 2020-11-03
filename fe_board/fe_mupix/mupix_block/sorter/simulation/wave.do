onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate /hitsorter_tb/reset_n
add wave -noupdate /hitsorter_tb/writeclk
add wave -noupdate /hitsorter_tb/tsclk
add wave -noupdate /hitsorter_tb/running
add wave -noupdate /hitsorter_tb/currentts
add wave -noupdate /hitsorter_tb/hit_in
add wave -noupdate /hitsorter_tb/hit_ena_in
add wave -noupdate /hitsorter_tb/dut/tscounter
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/running_last
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/running_read
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/running_seq
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/tslow
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/tshi
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/tsread
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/runstartup
add wave -noupdate -group Runcontrol /hitsorter_tb/dut/runshutdown
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/hit_last1
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/hit_last2
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/hit_last3
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/hit_ena_last1
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/hit_ena_last2
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/hit_ena_last3
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/tshit
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/slowtshit
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/sametsafternext
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/sametsnext
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/dcountertemp
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/dcountertemp2
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/tomem
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/memwren
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/waddr
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/tocmem
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/tocmem_hitwriter
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/fromcmem
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/cmemreadaddr
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/cmemwriteaddr
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/cmemreadaddr_hitwriter
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/cmemwriteaddr_hitwriter
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/cmemwren
add wave -noupdate -group {Hit writer} /hitsorter_tb/dut/cmemwren_hitwriter
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/reset
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/tofifo_counters
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/fromfifo_counters
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/read_counterfifo
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/write_counterfifo
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/counterfifo_almostfull
add wave -noupdate -group {Counter fifo} /hitsorter_tb/dut/counterfifo_empty
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/block_nonempty_accumulate
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/block_empty
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/block_empty_del1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/block_empty_del2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/stopwrite
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/stopwrite_del1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/stopwrite_del2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/blockchange
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/blockchange_del1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/blockchange_del2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_nnonempty
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_nechips
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_nechips2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_countchips
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_countchips_m1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_countchips_m2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/haseven
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_overflow
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_overflow_del1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/even_overflow_del2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_nnonempty
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_nechips
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_nechips2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_countchips
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_countchips_m1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_countchips_m2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/hasodd
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_overflow
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_overflow_del1
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/odd_overflow_del2
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/credits
add wave -noupdate -expand -group {Counter reader} /hitsorter_tb/dut/credittemp
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/state
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/current_block
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/fifo_reg
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/even_counters_reg
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/odd_counters_reg
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/doeven
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/doodd
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/subaddr
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/overflowts
add wave -noupdate -group Sequencer /hitsorter_tb/dut/seq/block_max
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/raddr
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/frommem
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/cmemreadaddr_hitreader
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/fromcmem_hitreader
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_last1
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_last2
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_last3
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_last4
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_ena
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_ena_last1
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_ena_last2
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_ena_last3
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/readcommand_ena_last4
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/outoverflow
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/overflow_last1
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/overflow_last2
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/overflow_last3
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/overflow_last4
add wave -noupdate -group {Output Generation} /hitsorter_tb/dut/memmultiplex
add wave -noupdate /hitsorter_tb/dut/data_out
add wave -noupdate /hitsorter_tb/dut/out_ena
add wave -noupdate /hitsorter_tb/dut/out_type
add wave -noupdate -group Diagnostics /hitsorter_tb/dut/noutoftime
add wave -noupdate -group Diagnostics /hitsorter_tb/dut/noverflow
add wave -noupdate -group Diagnostics /hitsorter_tb/dut/nintime
add wave -noupdate -group Diagnostics /hitsorter_tb/dut/nout
add wave -noupdate -group Diagnostics /hitsorter_tb/dut/diagnostic_sel
add wave -noupdate -group Diagnostics /hitsorter_tb/dut/diagnostic_out
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {4127596 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 367
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ns
update
WaveRestoreZoom {7582471 ps} {7813990 ps}
