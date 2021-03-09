library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.pcie_components.all;
use work.mudaq_registers.all;

entity pcie_register_mapping is
port (
    --! register inputs for pcie0
    i_pcie0_rregs_156   : in reg32array;
    i_pcie0_rregs_250   : in reg32array;

    --! register inputs for pcie1
    i_pcie1_rregs_156   : in reg32array;
    i_pcie1_rregs_250   : in reg32array;

    --! register outputs for pcie0/1
    o_pcie0_rregs       : out reg32array;
    o_pcie1_rregs       : out reg32array;

    -- slow 156 MHz clock
    i_clk_156           : in    std_logic;

    -- fast 250 MHz clock
    i_pcie0_clk         : in    std_logic;
    i_pcie1_clk         : in    std_logic--;
);
end entity;

--! @brief arch definition of the pcie_register_mapping
--! @details The arch of the pcie_register_mapping sync
--! the two clk domains used in the A10 board and outputs
--! two registers which are used for the two PCIe blocks
architecture arch of pcie_register_mapping is

    signal clk_sync_0, clk_sync_1 : std_logic;
    signal clk_last_0, clk_last_1 : std_logic;

begin

    --! sync read regs from slow (156.25 MHz) to fast (250 MHz) clock
    --! done for the first PCIe block
    process(i_pcie0_clk)
    begin
    if rising_edge(i_pcie0_clk) then
        clk_sync_0 <= i_clk_156;
        clk_last_0 <= clk_sync_0;
        
        if(clk_sync_0 = '1' and clk_last_0 = '0') then
            o_pcie0_rregs(PLL_REGISTER_R)                <= i_pcie0_rregs_156(PLL_REGISTER_R);
            o_pcie0_rregs(VERSION_REGISTER_R)            <= i_pcie0_rregs_156(VERSION_REGISTER_R);
            o_pcie0_rregs(RUN_NR_REGISTER_R)             <= i_pcie0_rregs_156(RUN_NR_REGISTER_R);
            o_pcie0_rregs(RUN_NR_ACK_REGISTER_R)         <= i_pcie0_rregs_156(RUN_NR_ACK_REGISTER_R);
            o_pcie0_rregs(RUN_STOP_ACK_REGISTER_R)       <= i_pcie0_rregs_156(RUN_STOP_ACK_REGISTER_R);
            o_pcie0_rregs(CNT_FEB_MERGE_TIMEOUT_R)       <= i_pcie0_rregs_156(CNT_FEB_MERGE_TIMEOUT_R);
            o_pcie0_rregs(CNT_FIFO_ALMOST_FULL_R)        <= i_pcie0_rregs_156(CNT_FIFO_ALMOST_FULL_R);
            o_pcie0_rregs(CNT_DC_LINK_FIFO_FULL_R)       <= i_pcie0_rregs_156(CNT_DC_LINK_FIFO_FULL_R);
            o_pcie0_rregs(CNT_SKIP_EVENT_LINK_FIFO_R)    <= i_pcie0_rregs_156(CNT_SKIP_EVENT_LINK_FIFO_R);
            o_pcie0_rregs(SC_MAIN_STATUS_REGISTER_R)     <= i_pcie0_rregs_156(SC_MAIN_STATUS_REGISTER_R);
            o_pcie0_rregs(MEM_WRITEADDR_HIGH_REGISTER_R) <= i_pcie0_rregs_156(MEM_WRITEADDR_HIGH_REGISTER_R);
            o_pcie0_rregs(MEM_WRITEADDR_LOW_REGISTER_R)  <= i_pcie0_rregs_156(MEM_WRITEADDR_LOW_REGISTER_R);
            o_pcie0_rregs(SC_STATE_REGISTER_R)           <= i_pcie0_rregs_156(SC_STATE_REGISTER_R);
        end if;
    end if;
    end process;

    --! map fast registers
    o_pcie0_rregs(DMA_STATUS_R)(DMA_DATA_WEN)   <= i_pcie0_rregs_250(DMA_STATUS_R)(DMA_DATA_WEN);
    o_pcie0_rregs(DMA_HALFFUL_REGISTER_R)       <= i_pcie0_rregs_250(DMA_HALFFUL_REGISTER_R);
    o_pcie0_rregs(DMA_NOTHALFFUL_REGISTER_R)    <= i_pcie0_rregs_250(DMA_NOTHALFFUL_REGISTER_R);
    o_pcie0_rregs(DMA_ENDEVENT_REGISTER_R)      <= i_pcie0_rregs_250(DMA_ENDEVENT_REGISTER_R);
    o_pcie0_rregs(DMA_NOTENDEVENT_REGISTER_R)   <= i_pcie0_rregs_250(DMA_NOTENDEVENT_REGISTER_R);
        

    --! sync read regs from slow (156.25 MHz) to fast (250 MHz) clock
    --! done for the second PCIe block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------


end architecture;
