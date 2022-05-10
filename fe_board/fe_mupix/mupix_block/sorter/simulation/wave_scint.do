onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/reset_n
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/writeclk
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/tsclk
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/running
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/currentts
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/hit_in
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/hit_ena_in
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/readclk
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/data_out
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/out_ena
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/out_type
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/diagnostic_out
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/counter
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/localts
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/clk156
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/running_last
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/running_read
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/running_read_last
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/running_seq
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/tslow
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/tshi
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/tsread
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/tsreadmemdelay
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/runstartup
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/runshutdown
add wave -noupdate -group {Run Control} -radix hexadecimal /hitsorter_scint_tb/dut/runend
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/hit_last1
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/hit_last2
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/hit_last3
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/hit_ena_last1
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/hit_ena_last2
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/hit_ena_last3
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/tshit
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/sametsafternext
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/sametsnext
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/dcountertemp
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/dcountertemp2
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/tomem
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/waddr
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/memwren
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/tocmem
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/tocmem_hitwriter
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/cmemwren_hitwriter
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/cmemwren
add wave -noupdate -group Writing -radix hexadecimal /hitsorter_scint_tb/dut/cmemwriteaddr
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/frommem
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/raddr
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/fromcmem
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/fromcmem_hitreader
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/cmemreadaddr
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/cmemreadaddr_hitwriter
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/cmemwriteaddr_hitwriter
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/cmemreadaddr_hitreader
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/addrcounterreset
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/reset
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/tofifo_counters
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/fromfifo_counters
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/read_counterfifo
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/write_counterfifo
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/counterfifo_almostfull
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/counterfifo_empty
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/running
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/running_last
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/stopped
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/output
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/current_block
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/current_ts
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/ts_to_out
add wave -noupdate -expand -group Sequencer -radix hexadecimal /hitsorter_scint_tb/dut/seq/counters_reg
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/subaddr
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/subaddr_to_out
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/chip_to_out
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/hasmem
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/hasoverflow
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/fifo_empty_last
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/fifo_new
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/line__95/copy_fifo
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/read_fifo_int
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/make_header
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/blockchange
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/no_copy_next
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/overflowts
add wave -noupdate -expand -group Sequencer /hitsorter_scint_tb/dut/seq/overflow_to_out
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/block_nonempty_accumulate
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/block_empty
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/block_empty_del1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/block_empty_del2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/stopwrite
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/stopwrite_del1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/stopwrite_del2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/blockchange
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/blockchange_del1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/blockchange_del2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_nnonempty
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_nechips
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_nechips2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_countchips
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_countchips_m1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_countchips_m2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/hashits
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_overflow
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_overflow_del1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/mem_overflow_del2
add wave -noupdate -radix decimal /hitsorter_scint_tb/dut/credits
add wave -noupdate -radix decimal /hitsorter_scint_tb/dut/line__533/creditchange
add wave -noupdate /hitsorter_scint_tb/dut/creditchange_reg
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/credits32
add wave -noupdate -radix decimal /hitsorter_scint_tb/dut/credittemp
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/hitcounter_sum_m3_mem
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/hitcounter_sum_mem
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/hitcounter_sum
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_last1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_last2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_last3
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_last4
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_ena
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_ena_last1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_ena_last2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_ena_last3
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/readcommand_ena_last4
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/outoverflow
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/overflow_last1
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/overflow_last2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/overflow_last3
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/overflow_last4
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/memmultiplex
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/tscounter
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/terminate_output
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/terminated_output
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/noutoftime
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/noverflow
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/nintime
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/nout
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/delay
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/noutoftime2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/noverflow2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/nintime2
add wave -noupdate -radix hexadecimal /hitsorter_scint_tb/dut/nout2
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 1} {14331439 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 583
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
WaveRestoreZoom {14133522 ps} {14343988 ps}
