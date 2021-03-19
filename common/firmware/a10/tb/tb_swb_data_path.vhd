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
    constant g_NLINKS_TOTL : integer := 64;
    constant g_NLINKS_FARM : integer := 8;
    constant g_NLINKS_DATA : integer := 12;

    signal clk, clk_fast, reset_n : std_logic := '0';
    --! data link signals
    signal rx : work.util.slv32_array_t(g_NLINKS_DATA-1 downto 0) := (others => (others => '0'));
    signal rx_k : work.util.slv4_array_t(g_NLINKS_DATA-1 downto 0) := (others => (others => '0'));

    signal writeregs_156 : work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));
    signal writeregs_250 : work.util.slv32_array_t(63 downto 0) := (others => (others => '0'));

    signal resets_n_156, resets_n_250 : std_logic_vector(31 downto 0) := (others => '0');

    signal counter : work.util.slv32_array_t(5+(g_NLINKS_DATA*3)-1 downto 0);
    
    signal farm_data : work.util.slv32_array_t(g_NLINKS_FARM - 1  downto 0);
    signal farm_datak : work.util.slv4_array_t(g_NLINKS_FARM - 1  downto 0);
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
    --! USE_GEN_LINK | USE_STREAM | USE_MERGER | USE_LINK | USE_GEN_MERGER | USE_FARM | SWB_READOUT_LINK_REGISTER_W | EFFECT                                                                         | WORKS 
    --! ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    --! 1            | 0          | 0          | 1        | 0              | 0        | n                           | Generate data for all 64 links, readout link n via DAM                         | x
    --! 1            | 1          | 0          | 0        | 0              | 0        | -                           | Generate data for all 64 links, simple merging of links, readout via DAM       | x
    --! 1            | 0          | 1          | 0        | 0              | 0        | -                           | Generate data for all 64 links, time merging of links, readout via DAM         | x
    --! 0            | 0          | 0          | 0        | 1              | 1        | -                           | Generate time merged data, send to farm                                        | x
    resets_n_156(RESET_BIT_DATAGEN)                             <= '0', '1' after (1.0 us / CLK_MHZ);
    writeregs_156(DATAGENERATOR_DIVIDER_REGISTER_W)             <= x"00000002";
    writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK)   <= '1';
    -- USE_GEN_LINK, USE_STREAM, USE_MERGER, USE_LINK, USE_GEN_MERGER, USE_FARM
    -- writeregs_250(SWB_READOUT_STATE_REGISTER_W)(5 downto 0)     <= "0100";
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM)     <= '0';
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER)     <= '1';
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_LINK)       <= '0';
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_MERGER) <= '0';
    writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_FARM)       <= '0';
        
    writeregs_250(SWB_LINK_MASK_PIXEL_REGISTER_W)               <= x"00000FFF";
    writeregs_250(SWB_READOUT_LINK_REGISTER_W)                  <= x"00000001";
    writeregs_250(GET_N_DMA_WORDS_REGISTER_W)                   <= (others => '1');
    writeregs_250(DMA_REGISTER_W)(DMA_BIT_ENABLE)               <= '1';
    mask_n <= x"00000000" & writeregs_250(SWB_LINK_MASK_PIXEL_REGISTER_W);


    --! SWB Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path : entity work.swb_data_path
    generic map (
        g_NLINKS_TOTL           => g_NLINKS_TOTL,
        g_NLINKS_FARM           => g_NLINKS_FARM,
        g_NLINKS_DATA           => g_NLINKS_DATA,
        LINK_FIFO_ADDR_WIDTH    => 8,
        TREE_w                  => 10,
        TREE_r                  => 10,
        SWB_ID                  => x"01",
        -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
        DATA_TYPE               => x"01"--;
    )
    port map(
        i_clk_156        => clk,
        i_clk_250        => clk_fast,
        
        i_reset_n_156    => reset_n,
        i_reset_n_250    => reset_n,

        i_resets_n_156   => resets_n_156,
        i_resets_n_250   => resets_n_250,
        
        i_rx             => rx,
        i_rx_k           => rx_k,
        i_rmask_n        => mask_n,

        i_writeregs_156  => writeregs_156,
        i_writeregs_250  => writeregs_250,

        o_counter        => counter,

        i_dmamemhalffull => '0',
        
        o_farm_data      => farm_data,
        o_farm_datak     => farm_datak,
        o_fram_wen       => fram_wen,

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
