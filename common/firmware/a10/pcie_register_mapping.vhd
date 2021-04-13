library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;


entity pcie_register_mapping is
port (
    --! register inputs for pcie0
    i_pcie0_rregs_156   : in    work.util.slv32_array_t(63 downto 0);
    i_pcie0_rregs_250   : in    work.util.slv32_array_t(63 downto 0);

    --! register inputs for pcie1
    i_pcie1_rregs_156   : in    work.util.slv32_array_t(63 downto 0);
    i_pcie1_rregs_250   : in    work.util.slv32_array_t(63 downto 0);
    
    --! register inputs for pcie0 from a10_block
    i_local_pcie0_rregs_156 : in    work.util.slv32_array_t(63 downto 0);
    i_local_pcie0_rregs_250 : in    work.util.slv32_array_t(63 downto 0);

    --! register outputs for pcie0/1
    o_pcie0_rregs       : out   reg32array_pcie;
    o_pcie1_rregs       : out   reg32array_pcie;

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

    signal rdempty : std_logic;
    signal data_rregs, q_rregs : std_logic_vector(64*32 - 1 downto 0);

begin

    --! sync read regs from slow (156.25 MHz) to fast (250 MHz) clock
    --! done for the first PCIe block
    gen_sync : FOR i in 0 to 63 GENERATE
        data_rregs(i * 32 + 31 downto i * 32) <= i_pcie0_rregs_156(i);
    END GENERATE gen_sync;

    --! sync slow registers
    e_sync_fifo : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 2, DATA_WIDTH  => 64*32--,
    ) port map ( data => data_rregs, wrreq => '1',
             rdreq => not rdempty, wrclk => i_clk_156, rdclk => i_pcie0_clk,
             q => q_rregs, rdempty => rdempty, aclr => '0'--,
    );
    
    --! map regs
    gen_map : FOR i in 0 to 63 GENERATE
        o_pcie0_rregs(i) <= i_local_pcie0_rregs_250(VERSION_REGISTER_R) when i = VERSION_REGISTER_R else
                            i_local_pcie0_rregs_250(DMA_STATUS_R) when i = DMA_STATUS_R else
                            i_local_pcie0_rregs_250(DMA_HALFFUL_REGISTER_R) when i = DMA_HALFFUL_REGISTER_R else
                            i_local_pcie0_rregs_250(DMA_NOTHALFFUL_REGISTER_R) when i = DMA_NOTHALFFUL_REGISTER_R else
                            i_local_pcie0_rregs_250(DMA_ENDEVENT_REGISTER_R) when i = DMA_ENDEVENT_REGISTER_R else
                            i_local_pcie0_rregs_250(DMA_NOTENDEVENT_REGISTER_R) when i = DMA_NOTENDEVENT_REGISTER_R else
                            q_rregs(i * 32 + 31 downto i * 32);
    END GENERATE gen_map;
    
    --! sync read regs from slow (156.25 MHz) to fast (250 MHz) clock
    --! done for the second PCIe block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------


end architecture;
