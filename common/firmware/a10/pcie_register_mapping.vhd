library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;


entity pcie_register_mapping is
port (
    --! register inputs for pcie0
    i_pcie0_rregs_A   : in    work.util.slv32_array_t(63 downto 0);
    i_pcie0_rregs_B   : in    work.util.slv32_array_t(63 downto 0);
    i_pcie0_rregs_C   : in    work.util.slv32_array_t(63 downto 0);
    
    --! register inputs for pcie0 from a10_block
    i_local_pcie0_rregs_A : in    work.util.slv32_array_t(63 downto 0);
    i_local_pcie0_rregs_B : in    work.util.slv32_array_t(63 downto 0);
    i_local_pcie0_rregs_C : in    work.util.slv32_array_t(63 downto 0);

    --! register outputs for pcie0
    o_pcie0_rregs       : out   reg32array_pcie;

    -- slow 156 MHz clock
    i_clk_A           : in    std_logic;    -- pcie clk
    i_clk_B           : in    std_logic;    -- link clk
    i_clk_C           : in    std_logic--;  -- ddr clk

);
end entity;

--! @brief arch definition of the pcie_register_mapping
--! @details The arch of the pcie_register_mapping sync
--! the two clk domains used in the A10 board and outputs
--! two registers which are used for the two PCIe blocks
architecture arch of pcie_register_mapping is

    signal rdempty_B, rdempty_C : std_logic;
    signal data_rregs_B, q_rregs_B, data_rregs_C, q_rregs_C : std_logic_vector(64 * 32 - 1 downto 0);

begin

    --! sync read regs from B, C clk to fast PCIe clock
    gen_sync : FOR i in 0 to 63 GENERATE
        data_rregs_B(i * 32 + 31 downto i * 32) <= i_pcie0_rregs_B(i);
        data_rregs_C(i * 32 + 31 downto i * 32) <= i_pcie0_rregs_C(i);
    END GENERATE gen_sync;

    --! sync FIFOs
    e_sync_fifo_B : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4, DATA_WIDTH  => 64*32--,
    ) port map ( data => data_rregs_B, wrreq => '1',
             rdreq => not rdempty_B, wrclk => i_clk_B, rdclk => i_clk_A,
             q => q_rregs_B, rdempty => rdempty_B, aclr => '0'--,
    );
    
    e_sync_fifo_C : entity work.ip_dcfifo
    generic map(
        ADDR_WIDTH  => 4, DATA_WIDTH  => 64*32--,
    ) port map ( data => data_rregs_C, wrreq => '1',
             rdreq => not rdempty_C, wrclk => i_clk_C, rdclk => i_clk_A,
             q => q_rregs_C, rdempty => rdempty_C, aclr => '0'--,
    );
    
    --! map regs
    gen_map : FOR i in 0 to 63 GENERATE
        o_pcie0_rregs(i) <= i_local_pcie0_rregs_A(VERSION_REGISTER_R) when i = VERSION_REGISTER_R else
                            i_local_pcie0_rregs_A(DMA_STATUS_R) when i = DMA_STATUS_R else
                            i_local_pcie0_rregs_A(DMA_HALFFUL_REGISTER_R) when i = DMA_HALFFUL_REGISTER_R else
                            i_local_pcie0_rregs_A(DMA_NOTHALFFUL_REGISTER_R) when i = DMA_NOTHALFFUL_REGISTER_R else
                            i_local_pcie0_rregs_A(DMA_ENDEVENT_REGISTER_R) when i = DMA_ENDEVENT_REGISTER_R else
                            i_local_pcie0_rregs_A(DMA_NOTENDEVENT_REGISTER_R) when i = DMA_NOTENDEVENT_REGISTER_R else
                            i_pcie0_rregs_A(EVENT_BUILD_STATUS_REGISTER_R) when i = EVENT_BUILD_STATUS_REGISTER_R else
                            i_pcie0_rregs_A(DMA_CNT_WORDS_REGISTER_R) when i = DMA_CNT_WORDS_REGISTER_R else
                            i_pcie0_rregs_A(SWB_COUNTER_REGISTER_R) when i = SWB_COUNTER_REGISTER_R else
                            i_pcie0_rregs_A(SWB_COUNTER_REGISTER_ADDR_R) when i = SWB_COUNTER_REGISTER_ADDR_R else
                            q_rregs_C(i * 32 + 31 downto i * 32) when i = DDR3_STATUS_R else
                            q_rregs_C(i * 32 + 31 downto i * 32) when i = DDR3_ERR_R else
                            q_rregs_C(i * 32 + 31 downto i * 32) when i = DATA_TSBLOCKS_R else
                            q_rregs_C(i * 32 + 31 downto i * 32) when i = DDR3_CLK_CNT_R else
                            q_rregs_B(i * 32 + 31 downto i * 32);
    END GENERATE gen_map;

end architecture;
