#

add_instance flash altera_onchip_flash
set_instance_parameter_value flash {CLOCK_FREQUENCY} [ expr 50000000 ]
set_instance_parameter_value flash {DATA_INTERFACE} {Parallel}

nios_base.connect flash clk nreset data 0x00000000
nios_base.connect flash ""    ""   csr 0x700F00F0

if { 1 } {
    add_connection cpu.instruction_master flash.data
    set_instance_parameter_value cpu {resetSlave} {flash.data}
}

# add fifo
add_instance flash_fifo altera_avalon_fifo
set_instance_parameter_value flash_fifo {fifoDepth} {512}
set_instance_parameter_value flash_fifo {useBackpressure} {1}
set_instance_parameter_value flash_fifo {singleClockMode} {0}
set_instance_parameter_value flash_fifo {useRegister} {0}
set_instance_parameter_value flash_fifo {useWriteControl} {1}
set_instance_parameter_value flash_fifo {useIRQ} {1}
set_instance_parameter_value flash_fifo {avalonMMAvalonMMDataWidth} {32}

add_connection clk.clk flash_fifo.clk_in
add_connection clk.clk_reset flash_fifo.reset_in
add_connection cpu.data_master flash_fifo.in
set_connection_parameter_value cpu.data_master/flash_fifo.in                      baseAddress {0x30000000}
add_connection cpu.data_master flash_fifo.in_csr
set_connection_parameter_value cpu.data_master/flash_fifo.in_csr                  baseAddress {0x40000000}
add_connection cpu.instruction_master flash_fifo.in_csr
add_connection cpu.irq flash_fifo.in_irq
set_interface_property clk_flash_fifo EXPORT_OF flash_fifo.clk_out
set_interface_property reset_flash_fifo EXPORT_OF flash_fifo.reset_out
set_interface_property out_flash_fifo EXPORT_OF flash_fifo.out

# add pio
nios_base.add_pio flash_ps_ctrl 32 Output 0x700F0600
nios_base.add_pio flash_w_cnt 32 Input 0x700F0640
nios_base.add_pio flash_cmd_addr 32 Output 0x700F0680
nios_base.add_pio flash_ctrl 8 Output 0x700F0720
nios_base.add_pio flash_status 8 Input 0x700F0760
nios_base.add_pio flash_o_data 8 Input 0x700F0800
nios_base.add_pio flash_i_data 8 Output 0x700F0840
