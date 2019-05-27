#**************************************************************
# This .sdc file is created by Terasic Tool.
# Users are recommended to modify this file to match users logic.
#**************************************************************

#**************************************************************
# Create Clock
#**************************************************************
create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2J]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B2L]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B3D]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B3F]
#create_clock -period  "50.000000 MHz" [get_ports CLK_50_B3H]
#create_clock -period "100.000000 MHz" [get_ports CLK_100_B3D]
#create_clock -period "100.000000 MHz" [get_ports CLKUSR_100]
#create_clock -period "266.667000 MHz" [get_ports DDR3A_REFCLK_p]
#create_clock -period "266.667000 MHz" [get_ports DDR3B_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIIA_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIIB_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIIC_REFCLK_p]
#create_clock -period "250.000000 MHz" [get_ports QDRIID_REFCLK_p]
create_clock -period "125.000000 MHz" [get_ports SMA_CLKIN]
create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]
#create_clock -period "156.250000 MHz" [get_ports refclk1_qr0_p]
#create_clock -period "125.000000 MHz" [get_ports refclk2_qr1_p]
#create_clock -period "250.000000 MHz" [get_ports QSFPB_REFCLK_p]
#create_clock -period "644.531250 MHz" [get_ports QSFPC_REFCLK_p]
#create_clock -period "644.531250 MHz" [get_ports QSFPD_REFCLK_p]
#create_clock -period "100.000000 MHz" [get_ports PCIE_REFCLK_p]
#create_clock -period "50 MHz" [get_ports SMA_CLKIN]
#set_false_path -from {pcie_block:pcie_b|pcie_application:pcie_app|dma_engine:dmaengine|dma_ram:dmaram|dma_ram_ram_2port_180_66q2v2i:ram_2port_0|altera_syncram:altera_syncram_component|altera_syncram_2ts1:auto_generated|altsyncram_ub94:altsyncram1|ram_block2a44~reg1} -to {pcie_block:pcie_b|pcie_application:pcie_app|dma_engine:dmaengine|tx_data_r[236]}

#**************************************************************
# Create Generated Clock
#**************************************************************
derive_pll_clocks -create_base_clocks
derive_clock_uncertainty



#**************************************************************
# Set Clock Latency
#**************************************************************



#**************************************************************
# Set Clock Uncertainty
#**************************************************************
derive_clock_uncertainty



#**************************************************************
# Set Input Delay
#**************************************************************



#**************************************************************
# Set Output Delay
#**************************************************************



#**************************************************************
# Set Clock Groups
#**************************************************************



#**************************************************************
# Set False Path
#**************************************************************
set_false_path -from [get_registers {reset_logic:resetlogic|resets_reg[*]}]
set_false_path -from {debouncer:deb1|dout_last}
set_false_path -from {debouncer:i_debouncer|q[0]}
set_false_path -from {debouncer:i_debouncer|q[0]} -to {sc_slave:sc_slave_ch0|state}

#**************************************************************
# Set Multicycle Path
#**************************************************************



#**************************************************************
# Set Maximum Delay
#**************************************************************



#**************************************************************
# Set Minimum Delay
#**************************************************************



#**************************************************************
# Set Input Transition
#**************************************************************



#**************************************************************
# Set Load
#**************************************************************



