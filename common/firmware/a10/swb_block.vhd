-------------------------------------------------------
--! @swb_block.vhd
--! @brief the swb_block can be used
--! for the LCHb Board and the development board
--! mainly it includes the datapath which includes
--! merging hits from multiple FEBs. There will be 
--! four types of SWB which differe accordingly to
--! the detector data they receive (inner pixel, 
--! scifi, down and up stream pixel/tiles)
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity swb_block is
generic (
    g_NLINKS_FEB_TOTL       : positive := 12;
    g_NLINKS_FARM_TOTL      : positive := 3;
    g_NLINKS_FARM_PIXEL     : positive := 2;
    g_NLINKS_DATA_PIXEL     : positive := 10;
    g_NLINKS_DATA_PIXEL_US  : positive := 5;
    g_NLINKS_DATA_PIXEL_DS  : positive := 5;
    g_NLINKS_FARM_SCIFI     : positive := 1;
    g_NLINKS_DATA_SCIFI     : positive := 2;
    SWB_ID                  : std_logic_vector(7 downto 0) := x"01"--;
);
port (

    --! links to/from FEBs
    -- TODO: rename to i_feb_data_rx, etc.
    i_rx                 : in  work.util.slv32_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    i_rx_k               : in  work.util.slv4_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    o_tx                 : out work.util.slv32_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    o_tx_k               : out work.util.slv4_array_t(g_NLINKS_FEB_TOTL-1 downto 0);

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

    o_farm_tx_data      : out   work.util.slv32_array_t(g_NLINKS_FARM_TOTL-1 downto 0);
    o_farm_tx_datak     : out   work.util.slv4_array_t(g_NLINKS_FARM_TOTL-1 downto 0);

    --! 250 MHz clock / reset_n
    i_reset_n_250        : in  std_logic;
    i_clk_250            : in  std_logic;    

    --! 156 MHz clock / reset_n
    i_reset_n_156        : in  std_logic;
    i_clk_156            : in  std_logic--;
);
end entity;

--! @brief arch definition of the swb_block
--! @details The arch of the swb_block can be used
--! for the LCHb Board and the development board
--! mainly it includes the datapath which includes
--! merging hits from multiple FEBs. There will be 
--! four types of SWB which differe accordingly to
--! the detector data they receive (inner pixel, 
--! scifi, down and up stream pixel/tiles)
architecture arch of swb_block is

    --! masking signals
    signal pixel_mask_n, scifi_mask_n : std_logic_vector(63 downto 0);
    
    --! farm links
    signal farm_data       : work.util.slv32_array_t(g_NLINKS_FARM_TOTL-1 downto 0);
    signal farm_data_valid : work.util.slv2_array_t(g_NLINKS_FARM_TOTL-1 downto 0);
    signal pixel_farm_data : work.util.slv32_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal scifi_farm_data : work.util.slv32_array_t(g_NLINKS_FARM_SCIFI-1 downto 0);
    signal pixel_farm_datak : work.util.slv4_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal scifi_farm_datak : work.util.slv4_array_t(g_NLINKS_FARM_SCIFI-1 downto 0);
    
    --! DMA
    signal pixel_dma_data : work.util.slv256_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal pixel_dma_cnt_words : work.util.slv32_array_t(g_NLINKS_FARM_PIXEL-1 downto 0);
    signal pixel_dma_wren, pixel_dma_endofevent, pixel_dma_done : std_logic_vector (g_NLINKS_FARM_PIXEL-1 downto 0);
    
    signal scifi_dma_data : std_logic_vector (255 downto 0);
    signal scifi_dma_wren, scifi_dma_endofevent, scifi_dma_done : std_logic;
    signal scifi_dma_cnt_words : std_logic_vector (31 downto 0);
    
    --! demerged FEB links
    signal rx_data         : work.util.slv32_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    signal rx_data_k       : work.util.slv4_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    signal rx_sc           : work.util.slv32_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    signal rx_sc_k         : work.util.slv4_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    signal rx_rc           : work.util.slv32_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    signal rx_rc_k         : work.util.slv4_array_t(g_NLINKS_FEB_TOTL-1 downto 0);
    signal rx_data_pixel   : work.util.slv32_array_t(g_NLINKS_DATA_PIXEL-1 downto 0);
    signal rx_data_k_pixel : work.util.slv4_array_t(g_NLINKS_DATA_PIXEL-1 downto 0);
    signal rx_data_scifi   : work.util.slv32_array_t(g_NLINKS_DATA_SCIFI-1 downto 0);
    signal rx_data_k_scifi : work.util.slv4_array_t(g_NLINKS_DATA_SCIFI-1 downto 0);
    
    --! counters
    signal counter_swb_data_pixel_156 : work.util.slv32_array_t(g_NLINKS_DATA_PIXEL*5-1 downto 0);
    signal counter_swb_data_scifi_156 : work.util.slv32_array_t(g_NLINKS_DATA_SCIFI*5-1 downto 0);
    signal counter_swb_data_pixel_250_us, counter_swb_data_pixel_250_ds, counter_swb_data_scifi_250 : work.util.slv32_array_t(5 downto 0);
    signal counter_swb_250 : work.util.slv32_array_t(17 downto 0);

begin

    --! @brief data path of the SWB board
    --! @details the data path of the SWB board is first splitting the 
    --! data from the FEBs into data, slow control and run control packages.
    --! The different paths are than assigned to the corresponding entities.
    --! The data is merged in time over all incoming FEBs. After this packages
    --! are build and the data is send of to the farm boars. The slow control
    --! data is saved in the PCIe memory and can be further used in the MIDAS 
    --! system. The run control packages are used to control the run and give 
    --! feedback to MIDAS if all FEBs started the run.

    --! counter readout
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- TODO: merger counters, sync to MIDAS
    e_counters : entity work.swb_readout_counters
    generic map (
        g_A_CNT             => 18,
        g_NLINKS_DATA_SCIFI => g_NLINKS_DATA_SCIFI,
        g_NLINKS_DATA_PIXEL => g_NLINKS_DATA_PIXEL--,
    )
    port map (
        --! register inputs for pcie0
        i_wregs_add_A       => i_writeregs_250(SWB_COUNTER_REGISTER_W),

        --! counters
        i_counter_A         => counter_swb_250,             -- pcie clk
        i_counter_B_pixel   => counter_swb_data_pixel_156,  -- link clk
        i_counter_B_scifi   => counter_swb_data_scifi_156,  -- link clk

        --! register outputs for pcie0
        o_pcie_data         => o_readregs_250(SWB_COUNTER_REGISTER_R),
        o_pcie_addr         => o_readregs_250(SWB_COUNTER_REGISTER_ADDR_R),

        --! i_reset
        i_reset_n_A         => i_reset_n_250,

        --! clocks
        i_clk_A             => i_clk_250,
        i_clk_B             => i_clk_156--,
    );


    --! demerge data
    --! three types of data will be extracted from the links
    --! data => detector data
    --! sc => slow control packages
    --! rc => runcontrol packages
    g_demerge: FOR i in g_NLINKS_FEB_TOTL-1 downto 0 GENERATE
        e_data_demerge : entity work.swb_data_demerger
        port map(
            i_clk               => i_clk_156,
            i_reset             => not i_resets_n_156(RESET_BIT_EVENT_COUNTER),
            i_aligned           => '1',
            i_data              => i_rx(i),
            i_datak             => i_rx_k(i),
            i_fifo_almost_full  => '0',--link_fifo_almost_full(i),
            o_data              => rx_data(i),
            o_datak             => rx_data_k(i),
            o_sc                => rx_sc(i),
            o_sck               => rx_sc_k(i),
            o_rc                => rx_rc(i),
            o_rck               => rx_rc_k(i),
            o_fpga_id           => open--,
        );
    end generate;


    --! run control used by MIDAS
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_run_control : entity work.run_control
    generic map (
        N_LINKS_g              => g_NLINKS_FEB_TOTL--,
    )
    port map (
        i_reset_ack_seen_n     => i_resets_n_156(RESET_BIT_RUN_START_ACK),
        i_reset_run_end_n      => i_resets_n_156(RESET_BIT_RUN_END_ACK),
        -- TODO: Write out padding 4kB at MIDAS Bank Builder if run end is done
        -- TODO: connect buffers emtpy from dma here
        -- o_all_run_end_seen => MIDAS Builder => i_buffer_empty
        i_buffers_empty        => (others => '1'),
        o_feb_merger_timeout   => o_readregs_156(CNT_FEB_MERGE_TIMEOUT_R),
        i_aligned              => (others => '1'),
        i_data                 => rx_rc,
        i_datak                => rx_rc_k,
        i_link_enable          => i_writeregs_156(FEB_ENABLE_REGISTER_W),
        i_addr                 => i_writeregs_156(RUN_NR_ADDR_REGISTER_W), -- ask for run number of FEB with this addr.
        i_run_number           => i_writeregs_156(RUN_NR_REGISTER_W)(23 downto 0),
        o_run_number           => o_readregs_156(RUN_NR_REGISTER_R), -- run number of i_addr
        o_runNr_ack            => o_readregs_156(RUN_NR_ACK_REGISTER_R), -- which FEBs have responded with run number in i_run_number
        o_run_stop_ack         => o_readregs_156(RUN_STOP_ACK_REGISTER_R),
        i_clk                  => i_clk_156--,
    );


    --! SWB slow control
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_sc_main : entity work.swb_sc_main
    generic map (
        NLINKS => g_NLINKS_FEB_TOTL--,
    )
    port map (
        i_clk           => i_clk_156,
        i_reset_n       => i_resets_n_156(RESET_BIT_SC_MAIN),
        i_length_we     => i_writeregs_156(SC_MAIN_ENABLE_REGISTER_W)(0),
        i_length        => i_writeregs_156(SC_MAIN_LENGTH_REGISTER_W)(15 downto 0),
        i_mem_data      => i_wmem_rdata,
        o_mem_addr      => o_wmem_addr,
        o_mem_data      => o_tx,
        o_mem_datak     => o_tx_k,
        o_done          => o_readregs_156(SC_MAIN_STATUS_REGISTER_R)(SC_MAIN_DONE),
        o_state         => o_readregs_156(SC_STATE_REGISTER_R)(27 downto 0)--,
    );
    
    e_sc_secondary : entity work.swb_sc_secondary
    generic map (
        NLINKS => g_NLINKS_FEB_TOTL--,
    )
    port map (
        reset_n                 => i_resets_n_156(RESET_BIT_SC_SECONDARY),
        i_link_enable           => i_writeregs_156(FEB_ENABLE_REGISTER_W)(g_NLINKS_FEB_TOTL-1 downto 0),
        link_data_in            => rx_sc,
        link_data_in_k          => rx_sc_k,
        mem_addr_out            => o_rmem_addr,
        mem_addr_finished_out   => o_readregs_156(MEM_WRITEADDR_LOW_REGISTER_R)(15 downto 0),
        mem_data_out            => o_rmem_wdata,
        mem_wren                => o_rmem_we,
        stateout                => o_readregs_156(SC_STATE_REGISTER_R)(31 downto 28),
        clk                     => i_clk_156--,
    );


    --! Mapping Signals
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- mask_n
    pixel_mask_n    <= x"00000000" & i_writeregs_250(SWB_LINK_MASK_PIXEL_REGISTER_W);
    scifi_mask_n    <= x"00000000" & i_writeregs_250(SWB_LINK_MASK_SCIFI_REGISTER_W);

    -- farm data
    o_farm_tx_data(g_NLINKS_FARM_PIXEL - 1 downto 0)                                          <= pixel_farm_data;
    o_farm_tx_datak(g_NLINKS_FARM_PIXEL - 1 downto 0)                                         <= pixel_farm_datak;
    o_farm_tx_data(g_NLINKS_FARM_PIXEL + g_NLINKS_FARM_SCIFI - 1 downto g_NLINKS_FARM_PIXEL)  <= scifi_farm_data;
    o_farm_tx_datak(g_NLINKS_FARM_PIXEL + g_NLINKS_FARM_SCIFI - 1 downto g_NLINKS_FARM_PIXEL) <= scifi_farm_datak;

    -- link mapping
    gen_pixel_data_mapping : FOR i in 0 to g_NLINKS_DATA_PIXEL - 1 GENERATE
        rx_data_pixel(i)   <= rx_data(i);
        rx_data_k_pixel(i) <= rx_data_k(i);
    END GENERATE gen_pixel_data_mapping;
    gen_scifi_data_mapping : FOR i in g_NLINKS_DATA_PIXEL to g_NLINKS_DATA_PIXEL + g_NLINKS_DATA_SCIFI - 1 GENERATE
        rx_data_scifi(i-g_NLINKS_DATA_PIXEL)   <= rx_data(i);
        rx_data_k_scifi(i-g_NLINKS_DATA_PIXEL) <= rx_data_k(i);
    END GENERATE gen_scifi_data_mapping;

    -- counter mapping
    counter_swb_250(5 downto 0) <= counter_swb_data_pixel_250_us;
    counter_swb_250(11 downto 6) <= counter_swb_data_pixel_250_ds;
    counter_swb_250(17 downto 12) <= counter_swb_data_scifi_250;

    -- DAM mapping
    o_dma_wren      <=  pixel_dma_wren(0) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                        pixel_dma_wren(1) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else
                        scifi_dma_wren    when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                        '0';
    o_endofevent    <=  pixel_dma_endofevent(0) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else 
                        pixel_dma_endofevent(1) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else 
                        scifi_dma_endofevent    when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                        '0';
    o_dma_data      <=  pixel_dma_data(0) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                        pixel_dma_data(1) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else 
                        scifi_dma_data    when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                        (others => '0');
    o_readregs_250(EVENT_BUILD_STATUS_REGISTER_R)(EVENT_BUILD_DONE) <=  pixel_dma_done(0) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                                                                        pixel_dma_done(1) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else
                                                                        scifi_dma_done    when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                                                                        '0';
    o_readregs_250(DMA_CNT_WORDS_REGISTER_R) <= pixel_dma_cnt_words(0) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_US) = '1' else
                                                pixel_dma_cnt_words(1) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_PIXEL_DS) = '1' else
                                                scifi_dma_cnt_words    when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_SCIFI) = '1' else
                                                (others => '0');

    --! SWB data path Pixel
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_data_path_pixel_us : entity work.swb_data_path
    generic map (
        g_LOOPUP_NAME           => "intRun2021",
        g_ADDR_WIDTH            => 11,
        g_NLINKS_DATA           => g_NLINKS_DATA_PIXEL_US,
        LINK_FIFO_ADDR_WIDTH    => 13,
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

        i_rx             => rx_data_pixel(g_NLINKS_DATA_PIXEL_US-1 downto 0),
        i_rx_k           => rx_data_k_pixel(g_NLINKS_DATA_PIXEL_US-1 downto 0),
        i_rmask_n        => pixel_mask_n(g_NLINKS_DATA_PIXEL_US-1 downto 0),

        i_writeregs_156  => i_writeregs_156,
        i_writeregs_250  => i_writeregs_250,

        o_counter_156    => counter_swb_data_pixel_156(g_NLINKS_DATA_PIXEL_US*5-1 downto 0),
        o_counter_250    => counter_swb_data_pixel_250_us,
        
        i_dmamemhalffull => i_dmamemhalffull,
        
        o_farm_data      => pixel_farm_data(0),
        o_farm_datak     => pixel_farm_datak(0),

        o_dma_wren       => pixel_dma_wren(0),
        o_dma_cnt_words  => pixel_dma_cnt_words(0),
        o_dma_done       => pixel_dma_done(0),
        o_endofevent     => pixel_dma_endofevent(0),
        o_dma_data       => pixel_dma_data(0)--;
    );
    
    e_swb_data_path_pixel_ds : entity work.swb_data_path
    generic map (
        g_LOOPUP_NAME           => "intRun2021",
        g_ADDR_WIDTH            => 11,
        g_NLINKS_DATA           => g_NLINKS_DATA_PIXEL_DS,
        LINK_FIFO_ADDR_WIDTH    => 13,
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

        i_rx             => rx_data_pixel(g_NLINKS_DATA_PIXEL-1 downto g_NLINKS_DATA_PIXEL_US),
        i_rx_k           => rx_data_k_pixel(g_NLINKS_DATA_PIXEL-1 downto g_NLINKS_DATA_PIXEL_US),
        i_rmask_n        => pixel_mask_n(g_NLINKS_DATA_PIXEL-1 downto g_NLINKS_DATA_PIXEL_US),

        i_writeregs_156  => i_writeregs_156,
        i_writeregs_250  => i_writeregs_250,

        o_counter_156    => counter_swb_data_pixel_156(g_NLINKS_DATA_PIXEL*5-1 downto g_NLINKS_DATA_PIXEL_US*5),
        o_counter_250    => counter_swb_data_pixel_250_ds,

        i_dmamemhalffull => i_dmamemhalffull,
        
        o_farm_data      => pixel_farm_data(1),
        o_farm_datak     => pixel_farm_datak(1),

        o_dma_wren       => pixel_dma_wren(1),
        o_dma_cnt_words  => pixel_dma_cnt_words(1),
        o_dma_done       => pixel_dma_done(1),
        o_endofevent     => pixel_dma_endofevent(1),
        o_dma_data       => pixel_dma_data(1)--;
    );


    --! SWB data path Scifi
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
--   e_swb_data_path_scifi : entity work.swb_data_path
--   generic map (
--       g_NLINKS_TOTL           => 64,
--       g_NLINKS_FARM           => g_NLINKS_FARM_SCIFI,
--       g_NLINKS_DATA           => g_NLINKS_DATA_SCIFI,
--       LINK_FIFO_ADDR_WIDTH    => 10,
--       TREE_w                  => 10,
--       TREE_r                  => 10,
--       SWB_ID                  => SWB_ID,
--       -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
--       DATA_TYPE               => x"02"--;
--   )
--   port map(
--        i_clk_156        => i_clk_156,
--        i_clk_250        => i_clk_250,
--
--        i_reset_n_156    => i_resets_n_156(RESET_BIT_DATA_PATH),
--        i_reset_n_250    => i_resets_n_250(RESET_BIT_DATA_PATH),
--
--        i_resets_n_156   => i_resets_n_156,
--        i_resets_n_250   => i_resets_n_250,
--
--        i_rx             => rx_data_scifi,
--        i_rx_k           => rx_data_k_scifi,
--        i_rmask_n        => scifi_mask_n(g_NLINKS_DATA_SCIFI-1 downto 0),
--
--        i_writeregs_156  => i_writeregs_156,
--        i_writeregs_250  => i_writeregs_250,
--
--        o_counter_156    => counter_swb_data_scifi_156,
--        o_counter_250    => counter_swb_data_scifi_250,
--
--        i_dmamemhalffull => i_dmamemhalffull,
--
--        o_farm_data      => scifi_farm_data,
--        o_farm_data_valid=> scifi_farm_data_valid,
--
--        o_dma_wren       => scifi_dma_wren,
--        o_dma_done       => scifi_dma_done,
--        o_endofevent     => scifi_dma_endofevent,
--        o_dma_data       => scifi_dma_data--;
--   );


    --! SWB data path Tile
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
--    e_swb_data_path : entity work.swb_data_path
--    generic map (
--        g_NLINKS_TOTL           => 64,
--        g_NLINKS_FARM           => 4,
--        g_NLINKS_DATA           => 2,
--        LINK_FIFO_ADDR_WIDTH    => 10,
--        TREE_w                  => 10,
--        TREE_r                  => 10,
--        SWB_ID                  => SWB_ID,
--        -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
--        DATA_TYPE               => x"03"--;
--    )
--    port map(
--        i_clk_156        => i_clk_156,
--        i_clk_250        => i_clk_250,
--        
--        i_reset_n_156    => i_resets_n_156(RESET_BIT_DATA_PATH),
--        i_reset_n_250    => i_resets_n_250(RESET_BIT_DATA_PATH),
--
--        i_resets_n_156   => i_resets_n_156,
--        i_resets_n_250   => i_resets_n_250,
--        
--        i_rx             => i_rx(15 downto 14),
--        i_rx_k           => i_rx_k(15 downto 14),
--        i_rmask_n        => x"0000000000" & i_writeregs_250(SWB_LINK_MASK_TILE_REGISTER_W),
--
--        i_writeregs_156  => i_writeregs_156,
--        i_writeregs_250  => i_writeregs_250,
--
--        o_counter        => counter_swb_data_tile,
--
--        i_dmamemhalffull => i_dmamemhalffull,
--        
--        o_farm_data      => o_tile_data,
--        o_farm_datak     => o_tile_datak,
--        o_fram_wen       => o_tile_wen,
--
--        o_dma_wren       => o_tile_dma_wren,
--        o_dma_done       => o_tile_dma_done,
--        o_endofevent     => o_tile_dma_endofevent,
--        o_dma_data       => o_tile_dma_data--;
--    );

end architecture;
