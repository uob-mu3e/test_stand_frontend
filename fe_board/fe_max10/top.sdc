# Create clocks
set_time_format -unit ns -decimal_places 3

create_clock -name {altera_reserved_tck} -period 100.000 -waveform {0.000 50.000} { altera_reserved_tck }
create_clock -name {max10_si_clk} -period 20.000 -waveform {0.000 10.000} { max10_si_clk }
create_clock -name {max10_osc_clk} -period 20.000 -waveform {0.000 10.000} { max10_osc_clk }
create_clock -name {max10_osc_clk} -period 20.000 -waveform {0.000 10.000} { max10_osc_clk }

derive_clock_uncertainty

derive_pll_clocks

set_clock_groups -asynchronous -group [get_clocks {altera_reserved_tck}] 

# SPI Input/Output delays flash spi
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {flash_io0}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {flash_io1}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {flash_io2}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {flash_io3}]

set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {flash_io0}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {flash_io1}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {flash_io2}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {flash_io3}]

set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {flash_sck}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {flash_csn}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {flash_io0}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {flash_io1}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {flash_io2}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {flash_io3}]

set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {flash_sck}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {flash_csn}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {flash_io0}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {flash_io1}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {flash_io2}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {flash_io3}]

# SPI Input/Output delays arria spi
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_clk}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_D1}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_D2}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_D3}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_csn}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_mosi}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 2 [get_ports {fpga_spi_miso}]

set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_clk}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_D1}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_D2}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_D3}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_csn}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_mosi}]
set_input_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 3 [get_ports {fpga_spi_miso}]



set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {fpga_spi_mosi}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {fpga_spi_miso}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {fpga_spi_D1}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {fpga_spi_D2}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -min 1 [get_ports {fpga_spi_D3}]

set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {fpga_spi_mosi}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {fpga_spi_miso}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {fpga_spi_D1}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {fpga_spi_D2}]
set_output_delay -clock { e_pll|altpll_component|auto_generated|pll1|clk[1] } -max 0 [get_ports {fpga_spi_D3}]


# Set False Path
set_false_path -from [get_clocks {max10_si_clk}] -to [get_clocks {max10_osc_clk}]
set_false_path -from [get_clocks {max10_osc_clk}] -to [get_clocks {max10_si_clk}]

set_false_path -to [get_keepers {*altera_std_synchronizer:*|din_s1}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[0]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[0]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[1]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[1]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[2]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[2]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[3]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[3]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[4]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[4]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[5]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[5]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[6]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[6]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[7]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[7]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[8]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[8]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[9]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[9]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[10]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[10]}]
set_false_path -from [get_keepers {*fiftyfivenm_adcblock_primitive_wrapper:adcblock_instance|wire_from_adc_dout[11]}] -to [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|dout_flp[11]}]
set_false_path -from [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|chsel[*]}] -to [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|chsel[*]}]
set_false_path -from [get_registers {*altera_modular_adc_control_fsm:u_control_fsm|soc}] -to [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|soc}]
set_false_path -to [get_registers {*|flash_busy_reg}]
set_false_path -to [get_registers {*|flash_busy_clear_reg}]
set_false_path -to [get_pins -nocase -compatibility_mode {*|alt_rst_sync_uq1|altera_reset_synchronizer_int_chain*|clrn}]
set_false_path -from [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_nios2_oci_break:the_nios_nios2_gen2_0_cpu_nios2_oci_break|break_readreg*}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_tck:the_nios_nios2_gen2_0_cpu_debug_slave_tck|*sr*}]
set_false_path -from [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_nios2_oci_debug:the_nios_nios2_gen2_0_cpu_nios2_oci_debug|*resetlatch}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_tck:the_nios_nios2_gen2_0_cpu_debug_slave_tck|*sr[33]}]
set_false_path -from [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_nios2_oci_debug:the_nios_nios2_gen2_0_cpu_nios2_oci_debug|monitor_ready}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_tck:the_nios_nios2_gen2_0_cpu_debug_slave_tck|*sr[0]}]
set_false_path -from [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_nios2_oci_debug:the_nios_nios2_gen2_0_cpu_nios2_oci_debug|monitor_error}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_tck:the_nios_nios2_gen2_0_cpu_debug_slave_tck|*sr[34]}]
set_false_path -from [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_nios2_ocimem:the_nios_nios2_gen2_0_cpu_nios2_ocimem|*MonDReg*}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_tck:the_nios_nios2_gen2_0_cpu_debug_slave_tck|*sr*}]
set_false_path -from [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_tck:the_nios_nios2_gen2_0_cpu_debug_slave_tck|*sr*}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_sysclk:the_nios_nios2_gen2_0_cpu_debug_slave_sysclk|*jdo*}]
set_false_path -from [get_keepers {sld_hub:*|irf_reg*}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_debug_slave_wrapper:the_nios_nios2_gen2_0_cpu_debug_slave_wrapper|nios_nios2_gen2_0_cpu_debug_slave_sysclk:the_nios_nios2_gen2_0_cpu_debug_slave_sysclk|ir*}]
set_false_path -from [get_keepers {sld_hub:*|sld_shadow_jsm:shadow_jsm|state[1]}] -to [get_keepers {*nios_nios2_gen2_0_cpu:*|nios_nios2_gen2_0_cpu_nios2_oci:the_nios_nios2_gen2_0_cpu_nios2_oci|nios_nios2_gen2_0_cpu_nios2_oci_debug:the_nios_nios2_gen2_0_cpu_nios2_oci_debug|monitor_go}]

# Set Net Delay
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|eoc}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|eoc}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|clk_dft}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|clk_dft}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[0]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[1]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[2]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[3]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[4]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[5]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[6]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[7]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[8]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[9]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[10]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[11]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[0]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[1]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[2]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[3]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[4]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[5]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[6]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[7]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[8]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[9]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[10]}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|adc_inst|adcblock_instance|primitive_instance|dout[11]}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|u_control_fsm|chsel[*]|q}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|u_control_fsm|chsel[*]|q}]
set_net_delay -max 5.000 -from [get_pins -compatibility_mode {*|u_control_fsm|soc|q}]
set_net_delay -min 0.000 -from [get_pins -compatibility_mode {*|u_control_fsm|soc|q}]
