library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.a10_pcie_registers.all;
use work.mudaq.all;

entity tb_swb_data_path is
end entity;

architecture arch of tb_swb_data_path is

    constant CLK_MHZ : real := 10000.0; -- MHz
    constant g_NLINKS_DATA : integer := 8;

    signal clk, clk_fast, reset_n : std_logic := '0';
    --! data link signals
    signal rx : work.mu3e.link_array_t(g_NLINKS_DATA-1 downto 0);

    signal writeregs : work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));

    signal resets_n : std_logic_vector(31 downto 0) := (others => '0');

    signal counter : work.util.slv32_array_t(5+(g_NLINKS_DATA*5)-1 downto 0);
    
    signal fram_wen, dma_wren, dma_done, endofevent : std_logic;
    signal dma_data : std_logic_vector (255 downto 0);
    signal mask_n : std_logic_vector(63 downto 0);

    signal dma_data_array : work.util.slv32_array_t(7 downto 0);

begin

    clk     <= not clk after (0.5 us / CLK_MHZ);
    clk_fast<= not clk_fast after (0.1 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);


    --! Setup
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! USE_GEN_LINK | USE_STREAM | USE_MERGER | USE_GEN_MERGER | USE_FARM | SWB_READOUT_LINK_REGISTER_W | EFFECT                                                                         | WORKS 
    --! ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --! 1            | 0          | 0          | 0              | 0        | n                           | Generate data for all 64 links, readout link n via DAM                         | x
    --! 1            | 1          | 0          | 0              | 0        | -                           | Generate data for all 64 links, simple merging of links, readout via DAM       | x
    --! 1            | 0          | 1          | 0              | 0        | -                           | Generate data for all 64 links, time merging of links, readout via DAM         | x
    --! 0            | 0          | 0          | 1              | 1        | -                           | Generate time merged data, send to farm                                        | x
    resets_n(RESET_BIT_DATAGEN)                             <= '0', '1' after (1.0 us / CLK_MHZ);
    resets_n(RESET_BIT_SWB_STREAM_MERGER)                   <= '0', '1' after (1.0 us / CLK_MHZ);
    resets_n(RESET_BIT_SWB_TIME_MERGER)                     <= '0', '1' after (1.0 us / CLK_MHZ);
    writeregs(DATAGENERATOR_DIVIDER_REGISTER_W)             <= x"00000002";
    writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK)   <= '1';
    writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_TEST_ERROR) <= '1'; 
    writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM)     <= '0';
    writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER)     <= '1';
    
    writeregs(SWB_LINK_MASK_PIXEL_REGISTER_W)               <= x"0000001F";--x"00000048";
    writeregs(SWB_READOUT_LINK_REGISTER_W)                  <= x"00000001";
    writeregs(GET_N_DMA_WORDS_REGISTER_W)                   <= (others => '1');
    writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE)               <= '1';
    mask_n <= x"00000000" & writeregs(SWB_LINK_MASK_PIXEL_REGISTER_W);


    --! SWB Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path : entity work.swb_data_path
    generic map (
        g_LOOPUP_NAME           => "intRun2021",
        g_ADDR_WIDTH            => 8,
        g_NLINKS_DATA           => g_NLINKS_DATA,
        LINK_FIFO_ADDR_WIDTH    => 8,
        SWB_ID                  => x"01",
        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
        DATA_TYPE               => "00"--;
    )
    port map(
        i_clk           => clk,
        i_reset_n       => reset_n,
        i_resets_n      => resets_n,
        
        i_rx            => rx,
            
        i_rmask_n       => mask_n(g_NLINKS_DATA - 1 downto 0),

        i_writeregs     => writeregs,

        o_counter       => open,

        i_dmamemhalffull => '0',
        
        o_farm_data      => open,

        o_dma_wren       => dma_wren,
        o_dma_done       => dma_done,
        o_endofevent     => endofevent,
        o_dma_data       => dma_data--;
    );

    dma_data_array(0) <= dma_data(0*32 + 31 downto 0*32);
    dma_data_array(1) <= dma_data(1*32 + 31 downto 1*32);
    dma_data_array(2) <= dma_data(2*32 + 31 downto 2*32);
    dma_data_array(3) <= dma_data(3*32 + 31 downto 3*32);
    dma_data_array(4) <= dma_data(4*32 + 31 downto 4*32);
    dma_data_array(5) <= dma_data(5*32 + 31 downto 5*32);
    dma_data_array(6) <= dma_data(6*32 + 31 downto 6*32);
    dma_data_array(7) <= dma_data(7*32 + 31 downto 7*32);

end architecture;
