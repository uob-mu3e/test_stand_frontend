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
    g_LOOPUP_NAME           : string    := "intRun2021";
    g_NLINKS_TOTL           : positive  := 3;
    g_ADDR_WIDTH            : positive  := 11;
    g_DDR4                  : boolean   := false;
    -- Data type: x"00" = pixel, x"01" = scifi, "10" = tiles
    DATA_TYPE               : std_logic_vector(1 downto 0) := "00";
    LINK_FIFO_ADDR_WIDTH    : positive  := 10--;
);
port (

    --! links to/from Farm
    i_rx                : in  work.mu3e.link_array_t(g_NLINKS_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    o_tx                : out work.mu3e.link_array_t(g_NLINKS_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    --! PCIe registers / memory
    i_writeregs         : in  work.util.slv32_array_t(63 downto 0);
    i_regwritten        : in  std_logic_vector(63 downto 0);
    o_readregs          : out work.util.slv32_array_t(63 downto 0);
    i_resets_n          : in  std_logic_vector(31 downto 0);


    i_dmamemhalffull    : in  std_logic;
    o_dma_wren          : out std_logic;
    o_endofevent        : out std_logic;
    o_dma_data          : out std_logic_vector(255 downto 0);

    --! 250 MHz clock pice / reset_n
    i_reset_n           : in  std_logic;
    i_clk               : in  std_logic;

    -- Interface to memory bank A
    o_A_mem_ck          : out   std_logic_vector(0 downto 0);                      -- mem_ck
    o_A_mem_ck_n        : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
    o_A_mem_a           : out   std_logic_vector(16 downto 0);                     -- mem_a
    o_A_mem_act_n       : out   std_logic_vector(0 downto 0);                      -- mem_act_n
    o_A_mem_ba          : out   std_logic_vector(2 downto 0);                      -- mem_ba
    o_A_mem_bg          : out   std_logic_vector(1 downto 0);                      -- mem_bg
    o_A_mem_cke         : out   std_logic_vector(0 downto 0);                      -- mem_cke
    o_A_mem_cs_n        : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
    o_A_mem_odt         : out   std_logic_vector(0 downto 0);                      -- mem_odt
    o_A_mem_reset_n     : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
    i_A_mem_alert_n     : in    std_logic_vector(0 downto 0)   := (others => 'X'); -- mem_alert_n
    o_A_mem_we_n        : out   std_logic_vector(0 downto 0);                      -- mem_we_n
    o_A_mem_ras_n       : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
    o_A_mem_cas_n       : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
    io_A_mem_dqs        : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
    io_A_mem_dqs_n      : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
    io_A_mem_dq         : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
    o_A_mem_dm          : out   std_logic_vector(7 downto 0);                      -- mem_dm
    io_A_mem_dbi_n      : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dbi_n
    i_A_oct_rzqin       : in    std_logic                      := 'X';             -- oct_rzqin
    i_A_pll_ref_clk     : in    std_logic                      := 'X';             -- clk

    -- Interface to memory bank B
    o_B_mem_ck          : out   std_logic_vector(0 downto 0);                      -- mem_ck
    o_B_mem_ck_n        : out   std_logic_vector(0 downto 0);                      -- mem_ck_n
    o_B_mem_a           : out   std_logic_vector(16 downto 0);                     -- mem_a
    o_B_mem_act_n       : out   std_logic_vector(0 downto 0);                      -- mem_act_n
    o_B_mem_ba          : out   std_logic_vector(2 downto 0);                      -- mem_ba
    o_B_mem_bg          : out   std_logic_vector(1 downto 0);                      -- mem_bg
    o_B_mem_cke         : out   std_logic_vector(0 downto 0);                      -- mem_cke
    o_B_mem_cs_n        : out   std_logic_vector(0 downto 0);                      -- mem_cs_n
    o_B_mem_odt         : out   std_logic_vector(0 downto 0);                      -- mem_odt
    o_B_mem_reset_n     : out   std_logic_vector(0 downto 0);                      -- mem_reset_n
    i_B_mem_alert_n     : in    std_logic_vector(0 downto 0)   := (others => 'X'); -- mem_alert_n
    o_B_mem_we_n        : out   std_logic_vector(0 downto 0);                      -- mem_we_n
    o_B_mem_ras_n       : out   std_logic_vector(0 downto 0);                      -- mem_ras_n
    o_B_mem_cas_n       : out   std_logic_vector(0 downto 0);                      -- mem_cas_n
    io_B_mem_dqs        : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs
    io_B_mem_dqs_n      : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dqs_n
    io_B_mem_dq         : inout std_logic_vector(63 downto 0)  := (others => 'X'); -- mem_dq
    o_B_mem_dm          : out   std_logic_vector(7 downto 0);                      -- mem_dm
    io_B_mem_dbi_n      : inout std_logic_vector(7 downto 0)   := (others => 'X'); -- mem_dbi_n
    i_B_oct_rzqin       : in    std_logic                      := 'X';             -- oct_rzqin
    i_B_pll_ref_clk     : in    std_logic                      := 'X'              -- clk

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
    signal rx : work.mu3e.link_array_t(g_NLINKS_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    --! data gen links
    signal gen_link : work.mu3e.link_t;

    --! aligned data
    signal aligned_data : work.mu3e.link_array_t(g_NLINKS_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);
    signal aligned_empty, aligned_ren : std_logic_vector(g_NLINKS_TOTL-1 downto 0);

    --! stream signal
    signal stream_rdata : work.mu3e.link_t;
    signal stream_counters : work.util.slv32_array_t(0 downto 0);
    signal stream_rempty, stream_ren : std_logic;
    signal stream_rack : std_logic_vector(g_NLINKS_TOTL - 1 downto 0);

    --! merger signal
    signal merger_rdata : work.mu3e.link_t;
    signal merger_rempty, merger_ren : std_logic;
    signal merger_rack : std_logic_vector(g_NLINKS_TOTL - 1 downto 0);

    --! ddr event builder
    signal ddr_rack, ddr_dma_wren, ddr_endofevent, ddr_dma_done : std_logic;
    signal ddr_dma_data : std_logic_vector(255 downto 0);
    signal ddr_data : std_logic_vector(511 downto 0);
    signal ddr_ts : std_logic_vector(47 downto 0);
    signal ddr_wen, ddr_ready, ddr_sop, ddr_eop, ddr_err : std_logic;

    --! debug event builder
    signal builder_data : work.mu3e.link_t;
    signal builder_rempty, builder_dma_wren, builder_endofevent, builder_dma_done, builder_rack : std_logic;
    signal builder_dma_data : std_logic_vector(255 downto 0);

    --! farm data path
    signal A_mem_ready      : std_logic;
    signal A_mem_calibrated : std_logic;
    signal A_mem_addr       : std_logic_vector(25 downto 0);
    signal A_mem_data       : std_logic_vector(511 downto 0);
    signal A_mem_write      : std_logic;
    signal A_mem_read       : std_logic;
    signal A_mem_q          : std_logic_vector(511 downto 0);
    signal A_mem_q_valid    : std_logic;
    signal B_mem_ready      : std_logic;
    signal B_mem_calibrated : std_logic;
    signal B_mem_addr       : std_logic_vector(25 downto 0);
    signal B_mem_data       : std_logic_vector(511 downto 0);
    signal B_mem_write      : std_logic;
    signal B_mem_read       : std_logic;
    signal B_mem_q          : std_logic_vector(511 downto 0);
    signal B_mem_q_valid    : std_logic;

    --! counters
    --! (g_NLINKS_TOTL*5)-1 downto 0 -> link to fifo counters
    --! (g_NLINKS_TOTL*4)+(g_NLINKS_TOTL*5)-1 downto (g_NLINKS_TOTL*5) -> link align counters
    signal counter_link_to_fifo : work.util.slv32_array_t((g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) - 1 downto 0);
    --! 0: bank_builder_idle_not_header
    --! 1: bank_builder_skip_event
    --! 2: bank_builder_cnt_event
    --! 3: bank_builder_tag_fifo_full
    signal counter_midas_event_builder : work.util.slv32_array_t(3 downto 0);
    --! 0: cnt_skip_event_dma
    --! 1: A_almost_full
    --! 2: B_almost_full
    --! 3: i_dmamemhalffull
    signal counter_ddr : work.util.slv32_array_t(3 downto 0);
    --! bank_builder_idle_not_header
    --! bank_builder_skip_event_dma
    --! bank_builder_ram_full
    --! bank_builder_tag_fifo_full
    signal builder_counters : work.util.slv32_array_t(3 downto 0);
    -- TODO: write status readout entity with ADDR to PCIe REGS and mapping to one counter REG
    signal counter : work.util.slv32_array_t(4 + 4 + 4 + (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) - 1 downto 0);

begin

    --! @brief data path of the Farm board
    --! @details the data path of the farm board is first aligning the
    --! data from the SWB and is than grouping them into Pixel, Scifi and Tiles.
    --! The data is saved according to the sub-header time in the DDR memory.
    --! Via MIDAS one can select how much data one wants to readout from the DDR memory
    --! the stored data is marked and than forworded to the next farm pc


    --! status counter / outputs
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    gen_link_to_fifo_cnt : FOR I in 0 to (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) - 1 GENERATE
        counter(I) <= counter_link_to_fifo(I);
    END GENERATE;
    gen_midas_event_cnt : FOR I in (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) to (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 3 GENERATE
        counter(I) <= counter_midas_event_builder(I-((g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5)));
    END GENERATE;
    gen_ddr_cnt : FOR I in (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 4 to (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 7 GENERATE
        counter(I) <= counter_ddr(I-((g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 4));
    END GENERATE;
    gen_builder_cnt : FOR I in (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 8 to (g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 11 GENERATE
        counter(I) <= builder_counters(I-((g_NLINKS_TOTL*4) + (g_NLINKS_TOTL*5) + 8));
    END GENERATE;

    --! SWB Data Generation
    --! generate data in the format from the SWB
    --! PIXEL US, PIXEL DS, SCIFI --> Int Run 2021
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- gen pixel data
    e_data_gen_link : entity work.data_generator_a10
    generic map (
        DATA_TYPE => DATA_TYPE,
        go_to_sh => 3,
        test_error => true,
        is_farm => true,
        go_to_trailer => 4--,
    )
    port map (
        i_enable            => i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK),
        i_seed              => (others => '1'),
        o_data              => gen_link,
        i_slow_down         => i_writeregs(DATAGENERATOR_DIVIDER_REGISTER_W),
        o_state             => open,

        i_reset_n           => i_resets_n(RESET_BIT_DATAGEN),
        i_clk               => i_clk--,
    );

    --! map links pixel / scifi
    gen_link_data : FOR I in 0 to g_NLINKS_TOTL - 1 GENERATE

        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n = '0' ) then
            rx(I)   <= work.mu3e.LINK_IDLE;
        elsif ( rising_edge( i_clk ) ) then
            if ( i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK) = '1' ) then
                rx(I)   <= work.mu3e.to_link(gen_link.data, gen_link.datak);
            else
                rx(I)   <= work.mu3e.to_link(i_rx(I).data, i_rx(I).datak);
            end if;
        end if;
        end process;

    END GENERATE gen_link_data;


    --! Link Alignment
    --! align data according to detector data
    --! two types of data will be extracted from the links
    --! PIXEL US, PIXEL DS, SCIFI --> Int Run 2021
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_link_to_fifo : entity work.farm_link_to_fifo
    generic map (
        g_LOOPUP_NAME       => g_LOOPUP_NAME,
        g_NLINKS_SWB_TOTL   => g_NLINKS_TOTL,
        LINK_FIFO_ADDR_WIDTH=> LINK_FIFO_ADDR_WIDTH--,
    )
    port map (
        --! link data
        i_rx                => rx,
        o_tx                => o_tx,

        --! data out
        o_data              => aligned_data,
        o_empty             => aligned_empty,
        i_ren               => aligned_ren,

        --! status counters
        --! (g_NLINKS_TOTL*5)-1 downto 0 -> link to fifo counters
        --! (g_NLINKS_TOTL*4)+(g_NLINKS_TOTL*5)-1 downto (g_NLINKS_TOTL*5) -> link align counters
        o_counter           => counter_link_to_fifo,

        i_clk               => i_clk,
        i_reset_n           => i_reset_n--,
    );


    --! stream merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_stream : entity work.swb_stream_merger
    generic map (
        g_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH,
        N => g_NLINKS_TOTL--,
    )
    port map (
        i_rdata     => aligned_data,
        i_rempty    => aligned_empty,
        i_rmask_n   => i_writeregs(FARM_LINK_MASK_REGISTER_W)(g_NLINKS_TOTL - 1 downto 0),
        o_rack      => stream_rack,

        -- farm data
        o_wdata     => stream_rdata,
        o_rempty    => stream_rempty,
        i_ren       => stream_ren,

        -- data for debug readout
        o_wdata_debug   => open,
        o_rempty_debug  => open,
        i_ren_debug     => '0',

        o_counters  => stream_counters,

        i_en        => i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM),
        i_reset_n   => i_resets_n(RESET_BIT_FARM_STREAM_MERGER),
        i_clk       => i_clk--,
    );


    --! time merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_time_merger : entity work.swb_time_merger
    generic map (
        g_ADDR_WIDTH    => g_ADDR_WIDTH,
        g_NLINKS_DATA   => g_NLINKS_TOTL,
        DATA_TYPE       => DATA_TYPE--,
    )
    port map (
        i_rx            => aligned_data,
        i_rempty        => aligned_empty,
        i_rmask_n       => i_writeregs(FARM_LINK_MASK_REGISTER_W)(g_NLINKS_TOTL - 1 downto 0),
        o_rack          => merger_rack,

        -- farm data
        o_wdata         => merger_rdata,
        o_rempty        => merger_rempty,
        i_ren           => merger_ren,

        -- data for debug readout
        o_wdata_debug   => open,
        o_rempty_debug  => open,
        i_ren_debug     => '0',

        o_error         => open,

        i_en            => i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER),
        i_reset_n       => i_resets_n(RESET_BIT_FARM_TIME_MERGER),
        i_clk           => i_clk--,
    );


    --! readout switches
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    aligned_ren     <=  stream_rack     when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rack     when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        (others => '0');
    builder_data    <=  stream_rdata    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rdata    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        work.mu3e.LINK_IDLE;
    builder_rempty  <=  stream_rempty   when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rempty   when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    stream_ren      <=  ddr_rack        when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' and i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '1' else
                        builder_rack    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' and i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '0' else
                        '0';
    merger_ren      <=  ddr_rack        when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' and i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '1' else
                        builder_rack    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' and i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '0' else
                        '0';
    o_dma_data      <=  ddr_dma_data        when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '1' else
                        builder_dma_data    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '0' else
                        (others => '0');
    o_dma_wren      <=  ddr_dma_wren        when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '1' else
                        builder_dma_wren    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '0' else
                        '0';
    o_endofevent    <=  ddr_endofevent      when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '1' else
                        builder_endofevent  when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '0' else
                        '0';
    o_readregs(EVENT_BUILD_STATUS_REGISTER_R)(EVENT_BUILD_DONE) <=  ddr_dma_done        when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '1' else
                                                                    builder_dma_done    when i_writeregs(FARM_READOUT_STATE_REGISTER_W)(USE_BIT_DDR) = '0' else
                                                                    '0';


    --! event builder used for the debug readout on the SWB
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_event_builder : entity work.swb_midas_event_builder
    port map (
        i_rx                => builder_data,
        i_rempty            => builder_rempty,

        i_get_n_words       => i_writeregs(GET_N_DMA_WORDS_REGISTER_W),
        i_dmamemhalffull    => i_dmamemhalffull,
        i_wen               => i_writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE),
        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
        i_data_type         => i_writeregs(FARM_DATA_TYPE_REGISTER_W)(FARM_DATA_TYPE_ADDR_RANGE),

        o_data              => builder_dma_data,
        o_wen               => builder_dma_wren,
        o_ren               => builder_rack,
        o_endofevent        => builder_endofevent,
        o_dma_cnt_words     => o_readregs(DMA_CNT_WORDS_REGISTER_R),
        o_done              => builder_dma_done,

        --! bank_builder_idle_not_header
        --! bank_builder_skip_event_dma
        --! bank_builder_ram_full
        --! bank_builder_tag_fifo_full
        o_counters          => builder_counters,

        i_reset_n           => i_reset_n,
        i_clk               => i_clk--,
    );


    --! Farm MIDAS Event Builder
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_midas_event_builder : entity work.farm_midas_event_builder_intrun22
    port map (
        --! data in
        i_rx            => builder_data,
        i_rempty        => builder_rempty,
        o_ren           => ddr_rack,

        -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
        i_data_type     => i_writeregs(FARM_DATA_TYPE_REGISTER_W)(FARM_DATA_TYPE_ADDR_RANGE),
        i_event_id      => i_writeregs(FARM_DATA_TYPE_REGISTER_W)(FARM_EVENT_ID_ADDR_RANGE),

        --! DDR data
        o_data          => ddr_data,
        o_wen           => ddr_wen,
        o_event_ts      => ddr_ts,
        i_ddr_ready     => ddr_ready,
        o_sop           => ddr_sop,
        o_eop           => ddr_eop,
        o_err           => ddr_err,

        --! status counters
        --! 0: bank_builder_idle_not_header
        --! 1: bank_builder_skip_event
        --! 2: bank_builder_cnt_event
        --! 3: bank_builder_tag_fifo_full
        o_counters      => counter_midas_event_builder,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );


    --! Farm Data Path
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_farm_data_path : entity work.farm_data_path
    port map(
        --! input from merging (first board) or links (subsequent boards)
        i_data           => ddr_data,
        i_data_en        => ddr_wen,
        i_ts             => ddr_ts(35 downto 4), -- 3:0 -> hit, 9:0 -> sub header
        o_ddr_ready      => ddr_ready,
        i_err            => ddr_err,
        i_sop            => ddr_sop,
        i_eop            => ddr_eop,

        --! input from PCIe demanding events
        i_ts_req_A        => i_writeregs(DATA_REQ_A_W),
        i_req_en_A        => i_regwritten(DATA_REQ_A_W),
        i_ts_req_B        => i_writeregs(DATA_REQ_B_W),
        i_req_en_B        => i_regwritten(DATA_REQ_B_W),
        i_tsblock_done    => i_writeregs(DATA_TSBLOCK_DONE_W)(15 downto 0),
        o_tsblocks        => o_readregs(DATA_TSBLOCKS_R),

        --! output to DMA
        o_dma_data       => ddr_dma_data,
        o_dma_wren       => ddr_dma_wren,
        o_dma_eoe        => ddr_endofevent,
        i_dmamemhalffull => i_dmamemhalffull,
        i_num_req_events => i_writeregs(FARM_REQ_EVENTS_W),
        o_dma_done       => ddr_dma_done,
        i_dma_wen        => i_writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE),

        --! status counters
        --! 0: cnt_skip_event_dma
        --! 1: A_almost_full
        --! 2: B_almost_full
        --! 3: i_dmamemhalffull
        o_counters      => counter_ddr,

        --! interface to memory bank A
        A_mem_ready     => A_mem_ready,
        A_mem_calibrated=> A_mem_calibrated,
        A_mem_addr      => A_mem_addr,
        A_mem_data      => A_mem_data,
        A_mem_write     => A_mem_write,
        A_mem_read      => A_mem_read,
        A_mem_q         => A_mem_q,
        A_mem_q_valid   => A_mem_q_valid,

        --! interface to memory bank B
        B_mem_ready     => B_mem_ready,
        B_mem_calibrated=> B_mem_calibrated,
        B_mem_addr      => B_mem_addr,
        B_mem_data      => B_mem_data,
        B_mem_write     => B_mem_write,
        B_mem_read      => B_mem_read,
        B_mem_q         => B_mem_q,
        B_mem_q_valid   => B_mem_q_valid,

        i_reset_n        => i_resets_n(RESET_BIT_DDR),
        i_clk            => i_clk--,
    );


    --! Farm DDR Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_ddr_block : entity work.ddr_block
    generic map (
        g_DDR4                => g_DDR4--,
    )
    port map(
        --! control and status registers
        i_ddr_control         => i_writeregs(DDR_CONTROL_W),

        --! A interface
        o_A_ddr_calibrated    => A_mem_calibrated,
        o_A_ddr_ready         => A_mem_ready,
        i_A_ddr_addr          => A_mem_addr,
        i_A_ddr_datain        => A_mem_data,
        o_A_ddr_dataout       => A_mem_q,
        i_A_ddr_write         => A_mem_write,
        i_A_ddr_read          => A_mem_read,
        o_A_ddr_read_valid    => A_mem_q_valid,

        --! B interface
        o_B_ddr_calibrated    => B_mem_calibrated,
        o_B_ddr_ready         => B_mem_ready,
        i_B_ddr_addr          => B_mem_addr,
        i_B_ddr_datain        => B_mem_data,
        o_B_ddr_dataout       => B_mem_q,
        i_B_ddr_write         => B_mem_write,
        i_B_ddr_read          => B_mem_read,
        o_B_ddr_read_valid    => B_mem_q_valid,

        --! error counters
        o_error               => o_readregs(DDR_ERR_R),

        --! interface to memory bank A
        o_A_mem_ck            => o_A_mem_ck,
        o_A_mem_ck_n          => o_A_mem_ck_n,
        o_A_mem_a             => o_A_mem_a,
        o_A_mem_act_n         => o_A_mem_act_n,
        o_A_mem_ba            => o_A_mem_ba,
        o_A_mem_bg            => o_A_mem_bg,
        o_A_mem_cke           => o_A_mem_cke,
        o_A_mem_cs_n          => o_A_mem_cs_n,
        o_A_mem_odt           => o_A_mem_odt,
        o_A_mem_reset_n(0)    => o_A_mem_reset_n(0),
        i_A_mem_alert_n(0)    => i_A_mem_alert_n(0),
        o_A_mem_we_n(0)       => o_A_mem_we_n(0),
        o_A_mem_ras_n(0)      => o_A_mem_ras_n(0),
        o_A_mem_cas_n(0)      => o_A_mem_cas_n(0),
        io_A_mem_dqs          => io_A_mem_dqs,
        io_A_mem_dqs_n        => io_A_mem_dqs_n,
        io_A_mem_dq           => io_A_mem_dq,
        o_A_mem_dm            => o_A_mem_dm,
        io_A_mem_dbi_n        => io_A_mem_dbi_n,
        i_A_oct_rzqin         => i_A_oct_rzqin,
        i_A_pll_ref_clk       => i_A_pll_ref_clk,

        --! interface to memory bank B
        o_B_mem_ck            => o_B_mem_ck,
        o_B_mem_ck_n          => o_B_mem_ck_n,
        o_B_mem_a             => o_B_mem_a,
        o_B_mem_act_n         => o_B_mem_act_n,
        o_B_mem_ba            => o_B_mem_ba,
        o_B_mem_bg            => o_B_mem_bg,
        o_B_mem_cke           => o_B_mem_cke,
        o_B_mem_cs_n          => o_B_mem_cs_n,
        o_B_mem_odt           => o_B_mem_odt,
        o_B_mem_reset_n(0)    => o_B_mem_reset_n(0),
        i_B_mem_alert_n(0)    => i_B_mem_alert_n(0),
        o_B_mem_we_n(0)       => o_B_mem_we_n(0),
        o_B_mem_ras_n(0)      => o_B_mem_ras_n(0),
        o_B_mem_cas_n(0)      => o_B_mem_cas_n(0),
        io_B_mem_dqs          => io_B_mem_dqs,
        io_B_mem_dqs_n        => io_B_mem_dqs_n,
        io_B_mem_dq           => io_B_mem_dq,
        o_B_mem_dm            => o_B_mem_dm,
        io_B_mem_dbi_n        => io_B_mem_dbi_n,
        i_B_oct_rzqin         => i_B_oct_rzqin,
        i_B_pll_ref_clk       => i_B_pll_ref_clk,

        i_reset_n             => i_resets_n(RESET_BIT_DDR),
        i_clk                 => i_clk--,
     );

end architecture;
