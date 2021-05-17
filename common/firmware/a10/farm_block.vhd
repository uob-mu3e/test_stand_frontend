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
    g_NLINKS_TOTL  : positive := 16;
    g_NLINKS_PIXEL : positive := 8;
    g_NLINKS_SCIFI : positive := 8;
    FARM_ID        : std_logic_vector(7 downto 0) := x"01"--;
);
port (

    --! links to/from FEBs
    i_rx                 : in  work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    i_rx_k               : in  work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    o_tx                 : out work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    o_tx_k               : out work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);

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
    
    --! SWB Data Generation
    --! generate data in the format from the SWB
    --! PIXEL, SCIFI --> Int Run 2021
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- gen pixel data
    e_data_gen_pixel : entity work.data_generator_merged_data
    generic map(
        go_to_sh => 2,
        go_to_trailer => 3--,
    )
    port map(
        i_clk       => i_clk_250_link,
        i_reset_n   => i_reset_n_250_link,
        i_en        => not full_pixel,
        i_sd        => x"00000002",
        o_data      => w_pixel,
        o_data_we   => w_pixel_en,
        o_state     => open--,
    );
    
    e_merger_fifo_pixel : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_pixel,
        wrreq           => w_pixel_en,
        rdreq           => r_pixel_en,
        clock           => i_clk_250_link,
        q               => r_pixel,
        full            => full_pixel,
        empty           => empty_pixel,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => not i_reset_n_250_link--,
    );
    
    e_swb_data_merger_pixel : entity work.swb_data_merger
    generic map (
        NLINKS      => NLINKS,
        DATA_TYPE   => x"01"--,
    )
    port map (
        i_reset_n   => i_reset_n_250_link,
        i_clk       => i_clk_250_link,
        
        i_data      => r_pixel,
        i_empty     => empty_pixel,
        
        o_ren       => r_pixel_en,
        o_wen       => open,
        o_data      => link_data_pixel,
        o_datak     => link_datak_pixel--,
    );
    
    -- gen scifi data
    e_data_gen_scifi : entity work.data_generator_merged_data
    generic map(
        go_to_sh => 2,
        go_to_trailer => 3--,
    )
    port map(
        i_clk       => dataclk,
        i_reset_n   => reset_n,
        i_en        => not full_scifi,
        i_sd        => x"00000002",
        o_data      => w_scifi,
        o_data_we   => w_scifi_en,
        o_state     => open--,
    );
    
    e_merger_fifo_scifi : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => NLINKS * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => w_scifi,
        wrreq           => w_scifi_en,
        rdreq           => r_scifi_en,
        clock           => dataclk,
        q               => r_scifi,
        full            => full_scifi,
        empty           => empty_scifi,
        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        sclr            => reset--,
    );
    
    e_swb_data_merger_scifi : entity work.swb_data_merger
    generic map (
        NLINKS      => NLINKS,
        DATA_TYPE   => x"02"--,
    )
    port map (
        i_reset_n   => reset_n,
        i_clk       => dataclk,
        
        i_data      => r_scifi,
        i_empty     => empty_scifi,
        
        o_ren       => r_scifi_en,
        o_wen       => open,
        o_data      => link_data_scifi,
        o_datak     => link_datak_scifi--,
    );
    
    -- map links pixel
    gen_map_links : FOR I in 0 to NLINKS - 1 GENERATE
        rx(I) <= link_data_pixel(I*32 + 31 downto I*32);
        rx_k(I) <= link_datak_pixel(I*4 + 3 downto I*4);
        rx(I+NLINKS) <= link_data_scifi(I*32 + 31 downto I*32);
        rx_k(I+NLINKS) <= link_datak_scifi(I*4 + 3 downto I*4);
    END GENERATE;
    

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
        N_SCIFI             => 8--,
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
        
        i_clk_250           => i_clk_250,  -- should be DDR clk
        i_reset_n_250       => i_resets_n_250(RESET_BIT_FARM_DATA_PATH)--,
    );
    

    --! Farm MIDAS Event Builder
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_midas_event_builder : entity work.farm_midas_event_builder
    generic map (
        g_NLINKS_SWB_TOTL => 16,
        N_PIXEL           => 8,
        N_SCIFI           => 8,
        RAM_ADDR          => 12--,
    )
    port map (
        i_pixel         => pixel_data,
        i_empty_pixel   => pixel_empty,
        o_ren_pixel     => pixel_ren,
    
        i_scifi         => scifi_data,
        i_empty_scifi   => scifi_empty,
        o_ren_scifi     => scifi_ren,

        -- DDR
        o_data          => data_in,
        o_wen           => data_wen,
        o_event_ts      => event_ts,
        i_ddr_ready     => ddr_ready,
    
        -- Link data
        o_pixel         => open,
        o_wen_pixel     => open,
    
        o_scifi         => open,
        o_wen_scifi     => open,

        o_counters      => open,

        i_reset_n_250   => reset_n,
        i_clk_250       => dataclk--,
    );
    
    
    --! Farm Data Path
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_data_path : entity work.farm_data_path 
    port map(
        reset_n         => reset_n,
        reset_n_ddr3    => reset_n,

        -- Input from merging (first board) or links (subsequent boards)
        dataclk         => dataclk,
        data_in         => data_in,
        data_en         => data_wen, 
        ts_in           => event_ts(35 downto 4), -- 3:0 -> hit, 9:0 -> sub header
        o_ddr_ready     => ddr_ready,

        -- Input from PCIe demanding events
        pcieclk         => pcieclk,
        ts_req_A        => ts_req_A,
        req_en_A        => req_en_A,
        ts_req_B        => ts_req_B,
        req_en_B        => req_en_B,
        tsblock_done    => tsblock_done,

        -- Output to DMA
        dma_data_out    => dma_data_out,
        dma_data_en     => dma_data_en,
        dma_eoe         => dma_eoe,
        i_dmamemhalffull=> '0',
        i_num_req_events=> ts_req_num,
        o_dma_done      => open,
        i_dma_wen       => '1',

        -- Interface to memory bank A
        A_mem_clk       => A_mem_clk,
        A_mem_ready     => A_mem_ready,
        A_mem_calibrated=> A_mem_calibrated,
        A_mem_addr      => A_mem_addr,
        A_mem_data      => A_mem_data,
        A_mem_write     => A_mem_write,
        A_mem_read      => A_mem_read,
        A_mem_q         => A_mem_q,
        A_mem_q_valid   => A_mem_q_valid,

        -- Interface to memory bank B
        B_mem_clk       => B_mem_clk,
        B_mem_ready     => B_mem_ready,
        B_mem_calibrated=> B_mem_calibrated,
        B_mem_addr		=> B_mem_addr,
        B_mem_data		=> B_mem_data,
        B_mem_write		=> B_mem_write,
        B_mem_read		=> B_mem_read,
        B_mem_q			=> B_mem_q,
        B_mem_q_valid	=> B_mem_q_valid
    );
    
    
    --! Farm DDR Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_ddr3_block : entity work.ddr3_block 
    port map(
        reset_n             => resets_n_ddr3(RESET_BIT_DDR3),
        
        -- Control and status registers
        ddr3control         => writeregs_ddr3(DDR3_CONTROL_W),
        ddr3status          => readregs_ddr3(DDR3_STATUS_R),

        -- A interface
        A_ddr3clk           => A_ddr3clk,
        A_ddr3calibrated    => A_ddr3calibrated,
        A_ddr3ready         => A_ddr3ready,
        A_ddr3addr          => A_ddr3addr,
        A_ddr3datain        => A_ddr3datain,
        A_ddr3dataout       => A_ddr3dataout,
        A_ddr3_write        => A_ddr3_write,
        A_ddr3_read         => A_ddr3_read,
        A_ddr3_read_valid   => A_ddr3_read_valid,
        
        -- B interface
        B_ddr3clk           => B_ddr3clk,
        B_ddr3calibrated    => B_ddr3calibrated,
        B_ddr3ready         => B_ddr3ready,
        B_ddr3addr          => B_ddr3addr,
        B_ddr3datain        => B_ddr3datain,
        B_ddr3dataout       => B_ddr3dataout,
        B_ddr3_write        => B_ddr3_write,
        B_ddr3_read         => B_ddr3_read,
        B_ddr3_read_valid   => B_ddr3_read_valid,
        
        -- Error counters
        errout              => readregs_ddr3(DDR3_ERR_R),

        -- Interface to memory bank A
        A_mem_ck            => DDR3A_CK,
        A_mem_ck_n          => DDR3A_CK_n,
        A_mem_a             => DDR3A_A,
        A_mem_ba            => DDR3A_BA,
        A_mem_cke           => DDR3A_CKE,
        A_mem_cs_n          => DDR3A_CS_n,
        A_mem_odt           => DDR3A_ODT,
        A_mem_reset_n(0)    => DDR3A_RESET_n,      
        A_mem_we_n(0)       => DDR3A_WE_n,
        A_mem_ras_n(0)      => DDR3A_RAS_n,
        A_mem_cas_n(0)      => DDR3A_CAS_n,
        A_mem_dqs           => DDR3A_DQS,
        A_mem_dqs_n         => DDR3A_DQS_n,
        A_mem_dq            => DDR3A_DQ,
        A_mem_dm            => DDR3A_DM,
        A_oct_rzqin         => RZQ_DDR3_A,
        A_pll_ref_clk       => DDR3A_REFCLK_p,
        
        -- Interface to memory bank B
        B_mem_ck            => DDR3B_CK,
        B_mem_ck_n          => DDR3B_CK_n,
        B_mem_a             => DDR3B_A,
        B_mem_ba            => DDR3B_BA,
        B_mem_cke           => DDR3B_CKE,
        B_mem_cs_n          => DDR3B_CS_n,
        B_mem_odt           => DDR3B_ODT,
        B_mem_reset_n(0)    => DDR3B_RESET_n,      
        B_mem_we_n(0)       => DDR3B_WE_n,
        B_mem_ras_n(0)      => DDR3B_RAS_n,
        B_mem_cas_n(0)      => DDR3B_CAS_n,
        B_mem_dqs           => DDR3B_DQS,
        B_mem_dqs_n         => DDR3B_DQS_n,
        B_mem_dq            => DDR3B_DQ,
        B_mem_dm            => DDR3B_DM,
        B_oct_rzqin         => RZQ_DDR3_B,
        B_pll_ref_clk       => DDR3B_REFCLK_p--,
     );
 
     DDR3A_SDA   <= 'Z';
     DDR3B_SDA   <= 'Z';

end architecture;
