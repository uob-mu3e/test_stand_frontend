#force standard values for clock divider,emulators 
#script parameter $1: path of DUT to force, e.g. "testbench/dut"
force -deposit sim:/$1/u_digital_all/u_clk_div_sys_clk/bit_counter 0000 0

### force values for vhdl #{{{
### for receive_all mode, reset the reg1 to 0 #{{{
for {set i 0} {$i <4} {incr i} {
	for {set j 0} {$j <8} {incr j} {
		force -deposit sim:/$1/u_digital_all/L1_group_buf_gen($i)/u_L1_buffer/ch_recv_gen($j)/ch_rcv/data_reg1 0000000000000000000000000000000000000000000000 0
	}
}
#}}}

### define the CHANNEL_NUMBER and seed for tdc emulators  #{{{
for {set i 0} {$i < 16} {incr i} {
	force -freeze sim:/$1/gen_tdcs_left_bottom($i)/u_tdc/CHANNEL_NUMBER $i 0
	force -deposit sim:/$1/gen_tdcs_left_bottom($i)/u_tdc/s_seed1 [expr $i+1+32*$2] 0
	force -freeze sim:/$1/gen_tdcs_left_bottom($i)/u_tdc/ASIC_NUMBER $2 0
}
for {set i 16} {$i < 32} {incr i} {
	force -freeze sim:/$1/gen_tdcs_left_top($i)/u_tdc/CHANNEL_NUMBER $i 0
	force -deposit sim:/$1/gen_tdcs_left_top($i)/u_tdc/s_seed1 [expr $i+1+32*$2] 0
	force -freeze sim:/$1/gen_tdcs_left_top($i)/u_tdc/ASIC_NUMBER $2 0
}

#}}}


