-------------------------------------------------------
--! @farm_block.vhd
--! @brief the farm_block can be used
--! for the development board mainly it includes 
--! the datapath which includes merging detector data
--! from multiple SWBs.
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity farm_block is
generic (
    g_NLINKS_SWB_TOTL   : positive := 16;
    g_NLINKS_FARM_TOTL  : positive := 16;
    g_NLINKS_FARM_PIXEL : positive := 8;
    g_NLINKS_DATA_PIXEL : positive := 12;
    SWB_ID              : std_logic_vector(7 downto 0) := x"01"--;
);
port (

    --! links to/from FEBs
    i_rx                 : in  work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    i_rx_k               : in  work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    o_tx                 : out work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    o_tx_k               : out work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);

    --! PCIe registers / memory
    i_writeregs_250      : in  work.util.slv32_array_t(63 downto 0);
    i_writeregs_156      : in  work.util.slv32_array_t(63 downto 0);
    
    o_readregs_250       : out work.util.slv32_array_t(63 downto 0);
    o_readregs_156       : out work.util.slv32_array_t(63 downto 0);

    i_resets_n_250       : in  std_logic_vector(31 downto 0);
    i_resets_n_156       : in  std_logic_vector(31 downto 0);

    i_wmem_rdata         : in  std_logic_vector(31 downto 0);
    o_wmem_addr          : out std_logic_vector(15 downto 0);

    o_rmem_wdata         : out std_logic_vector(31 downto 0);
    o_rmem_addr          : out std_logic_vector(15 downto 0);
    o_rmem_we            : out std_logic;

    i_dmamemhalffull     : in  std_logic;
    o_dma_wren           : out std_logic;
    o_endofevent         : out std_logic;
    o_dma_data           : out std_logic_vector(255 downto 0);
    
    --! 250 MHz clock pice / reset_n
    i_reset_n_250        : in  std_logic;
    i_clk_250            : in  std_logic;   

    --! 250 MHz clock link / reset_n
    i_reset_n_250_link   : in  std_logic;
    i_clk_250_link       : in  std_logic;    

    --! 156 MHz clock / reset_n
    i_reset_n_156        : in  std_logic;
    i_clk_156            : in  std_logic--;
);
end entity;

--! @brief arch definition of the farm_block
--! @details the farm_block can be used
--! for the development board mainly it includes 
--! the datapath which includes merging detector data
--! from multiple SWBs.
--! scifi, down and up stream pixel/tiles)
architecture arch of farm_block is

    --! mapping signals
    --! fiber link_mapping(0)=1 
    --! Fiber QSFPA.1 is mapped to first(0) link
    type mapping_t is array(natural range <>) of integer;
    constant NLINKS_DATA : integer := 3;
    constant link_mapping : mapping_t(NLINKS_DATA-1 downto 0) := (1,2,4);
    signal pixel_mask_n : std_logic_vector(63 downto 0);
    
    --! farm links
    signal pixel_farm_data : work.util.slv32_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal pixel_farm_datak : work.util.slv4_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal pixel_fram_wen : std_logic;
    
    --! DMA
    signal pixel_dma_data : std_logic_vector (255 downto 0);
    signal pixel_dma_cnt_words : std_logic_vector (31 downto 0);
    signal pixel_dma_wren, pixel_dma_endofevent, pixel_dma_done : std_logic;
    
    --! demerged FEB links
    signal rx_data         : work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    signal rx_data_k       : work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    signal rx_sc           : work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    signal rx_sc_k         : work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    signal rx_rc           : work.util.slv32_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    signal rx_rc_k         : work.util.slv4_array_t(g_NLINKS_SWB_TOTL-1 downto 0);
    signal rx_data_pixel   : work.util.slv32_array_t(g_NLINKS_DATA_PIXEL-1 downto 0);
    signal rx_data_k_pixel : work.util.slv4_array_t(g_NLINKS_DATA_PIXEL-1 downto 0);
    
    --! counters
    signal counter_swb_data_pixel : work.util.slv32_array_t(5+(g_NLINKS_DATA_PIXEL*3)-1 downto 0);


begin

    --! @brief data path of the Farm board
    --! @details the data path of the farm board is first aligning the 
    --! data from the SWB and is than grouping them into Pixel, Scifi and Tiles.
    --! The data is saved according to the sub-header time in the DDR3 memory.
    --! Via MIDAS one can select how much data one wants to readout from the DDR3 memory
    --! the stored data is marked and than forworded to the next farm pc

    --! Link Mapping
    --! align data according to detector data
    --! two types of data will be extracted from the links
    --! PIXEL, SCIFI --> Int Run 2021
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_demerge_pixel : entity work.farm_link_to_fifo
    generic map (
        g_NLINKS_SWB_TOTL   => g_NLINKS_SWB_TOTL,
        N_PIXEL             => 8,
        N_SCIFI             => 8,
    )
    port map (
        i_rx                => i_rx,
        i_rx_k              => i_rx_k,

        -- pixel data
        o_pixel             => open,
        o_empty_pixel       => open, 
        i_ren_pixel         => '1',
        o_error_pixel       => open,
        
        -- scifi data
        o_scifi             => open,
        o_empty_scifi       => open,
        i_ren_scifi         => '1',
        o_error_scifi       => open,
    
        --! error counters 
        --! 0: fifo f_almost_full
        --! 1: fifo f_wrfull
        --! 2: # of skip event
        --! 3: # of events
        o_counter           => open, -- out work.util.slv32_array_t(4 * g_NLINKS_SWB_TOTL - 1 downto 0);
        
        i_clk_250_link      => i_clk_250_link,
        i_reset_n_250_link  => i_reset_n_250_link,
        
        i_clk_250           => i_clk_250,
        i_reset_n_250       => not i_resets_n_250(RESET_BIT_FARM_DATA_PATH)--,
    );
    

    --! SWB data path Pixel
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path_pixel : entity work.swb_data_path
    generic map (
        g_NLINKS_TOTL           => 64,
        g_NLINKS_FARM           => g_NLINKS_FARM_PIXEL,
        g_NLINKS_DATA           => g_NLINKS_DATA_PIXEL,
        LINK_FIFO_ADDR_WIDTH    => 10,
        TREE_w                  => 10,
        TREE_r                  => 10,
        SWB_ID                  => SWB_ID,
        -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
        DATA_TYPE               => x"01"--;
    )
    port map(
        i_clk_156        => i_clk_156,
        i_clk_250        => i_clk_250,
        
        i_reset_n_156    => i_resets_n_156(RESET_BIT_DATA_PATH),
        i_reset_n_250    => i_resets_n_250(RESET_BIT_DATA_PATH),

        i_resets_n_156   => i_resets_n_156,
        i_resets_n_250   => i_resets_n_250,

        --TODO: do link mapping
        i_rx             => rx_data_pixel,
        i_rx_k           => rx_data_k_pixel,
        i_rmask_n        => pixel_mask_n,

        i_writeregs_156  => i_writeregs_156,
        i_writeregs_250  => i_writeregs_250,

        o_counter        => counter_swb_data_pixel,

        i_dmamemhalffull => i_dmamemhalffull,
        
        o_farm_data      => pixel_farm_data,
        o_farm_datak     => pixel_farm_datak,
        o_fram_wen       => pixel_fram_wen,

        o_dma_wren       => pixel_dma_wren,
        o_dma_cnt_words  => pixel_dma_cnt_words,
        o_dma_done       => pixel_dma_done,
        o_endofevent     => pixel_dma_endofevent,
        o_dma_data       => pixel_dma_data--;
    );

end architecture;
