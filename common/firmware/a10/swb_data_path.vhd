-------------------------------------------------------
--! @swb_data_path.vhd
--! @brief the swb_data_path can be used
--! for the LCHb Board and the development board
--! mainly it includes the datapath which includes
--! merging hits from multiple FEBs.
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.a10_pcie_registers.all;
use work.mudaq.all;


entity swb_data_path is
generic (
    g_LOOPUP_NAME : string := "intRun2021";
    g_ADDR_WIDTH : positive := 11;
    g_NLINKS_DATA : positive := 8;
    LINK_FIFO_ADDR_WIDTH : positive := 10;
    SWB_ID : std_logic_vector(7 downto 0) := x"01";
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE : std_logic_vector(7 downto 0) := x"01"--;
);
port (
    i_resets_n_156   : in  std_logic_vector(31 downto 0);
    i_resets_n_250   : in  std_logic_vector(31 downto 0);

    i_rx             : in  work.util.slv32_array_t(g_NLINKS_DATA-1 downto 0);
    i_rx_k           : in  work.util.slv4_array_t(g_NLINKS_DATA-1 downto 0);
    i_rmask_n        : in  std_logic_vector(g_NLINKS_DATA-1 downto 0);

    i_writeregs_156  : in  work.util.slv32_array_t(63 downto 0);
    i_writeregs_250  : in  work.util.slv32_array_t(63 downto 0);

    o_counter_156    : out work.util.slv32_array_t(g_NLINKS_DATA*5-1 downto 0);
    o_counter_250    : out work.util.slv32_array_t(5 downto 0);

    i_dmamemhalffull : in  std_logic;

    o_farm_data      : out std_logic_vector (31 downto 0);
    o_farm_datak     : out std_logic_vector (3 downto 0);

    o_dma_wren       : out std_logic;
    o_dma_cnt_words  : out std_logic_vector (31 downto 0);
    o_dma_done       : out std_logic;
    o_endofevent     : out std_logic;
    o_dma_data       : out std_logic_vector (255 downto 0);

    i_reset_n_156       : in    std_logic;
    i_reset_n_250       : in    std_logic;
    i_clk_156           : in    std_logic;
    i_clk_250           : in    std_logic--;
);
end entity;

architecture arch of swb_data_path is

    signal reset_250_n : std_logic;

    --! data gen links
    signal gen_link, gen_link_error : std_logic_vector(31 downto 0);
    signal gen_link_k, gen_link_k_error : std_logic_vector(3 downto 0);

    --! data link signals
    signal rx : work.util.slv32_array_t(g_NLINKS_DATA-1 downto 0);
    signal rx_k : work.util.slv4_array_t(g_NLINKS_DATA-1 downto 0);
    signal rx_ren, rx_mask_n, rx_rdempty : std_logic_vector(g_NLINKS_DATA-1 downto 0) := (others => '0');
    signal rx_q : work.util.slv35_array_t(g_NLINKS_DATA-1 downto 0) := (others => (others => '0'));
    signal rx_q_s : work.util.slv32_array_t(g_NLINKS_DATA-1 downto 0) := (others => (others => '0'));
    signal sop, eop, shop, t0, t1, hit : std_logic_vector(g_NLINKS_DATA-1 downto 0) := (others => '0');

    --! stream merger
    signal stream_rdata, stream_rdata_debug : std_logic_vector(31 downto 0);
    signal stream_counters : work.util.slv32_array_t(0 downto 0);
    signal stream_rempty, stream_ren, stream_header, stream_trailer : std_logic;
    signal stream_rempty_debug, stream_ren_debug, stream_header_debug, stream_trailer_debug : std_logic;
    signal stream_rack : std_logic_vector(g_NLINKS_DATA-1 downto 0);

    --! timer merger
    signal merger_rdata : std_logic_vector(31 downto 0);
    signal merger_rdata_debug : std_logic_vector(31 downto 0);
    signal merger_rempty, merger_ren, merger_header, merger_trailer, merger_error : std_logic;
    signal merger_rempty_debug, merger_ren_debug, merger_header_debug, merger_trailer_debug, merger_error_debug : std_logic;
    signal merger_rack : std_logic_vector (g_NLINKS_DATA-1 downto 0);

    --! event builder
    signal builder_data : std_logic_vector(31 downto 0);
    signal builder_counters : work.util.slv32_array_t(3 downto 0);
    signal builder_rempty, builder_rack, builder_header, builder_trailer, builder_error : std_logic;

    --! links to farm
    signal farm_data : std_logic_vector(31 downto 0);
    signal farm_rack, farm_rempty, farm_header, farm_trailer : std_logic;

    --! status counters
    signal link_to_fifo_cnt : work.util.slv32_array_t((g_NLINKS_DATA*5)-1 downto 0);
    signal events_to_farm_cnt : std_logic_vector(31 downto 0);

begin

        --! generate reset for 250 MHz
    e_reset_250_n : entity work.reset_sync
    port map ( o_reset_n => reset_250_n, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );

    --! status counter
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! TODO: add this to counters
    -- tag_fifo_empty;
    -- dma_write_state;
    -- rx_rdempty;

    -- 250 MHz counters
    o_counter_250(0) <= stream_counters(0);  --! e_stream_fifo full
    o_counter_250(1) <= builder_counters(0); --! bank_builder_idle_not_header
    o_counter_250(2) <= builder_counters(1); --! bank_builder_skip_event_dma
    o_counter_250(3) <= builder_counters(2); --! bank_builder_ram_full
    o_counter_250(4) <= builder_counters(3); --! bank_builder_tag_fifo_full
    o_counter_250(5) <= events_to_farm_cnt; --! events send to the farm

    -- 156 MHz counters
    generate_rdata : for i in 0 to g_NLINKS_DATA - 1 generate
        o_counter_156(0+i*5) <= link_to_fifo_cnt(0+i*5); --! fifo almost_full
        o_counter_156(1+i*5) <= link_to_fifo_cnt(1+i*5); --! fifo wrfull
        o_counter_156(2+i*5) <= link_to_fifo_cnt(2+i*5); --! # of skip event
        o_counter_156(3+i*5) <= link_to_fifo_cnt(3+i*5); --! # of events
        o_counter_156(4+i*5) <= link_to_fifo_cnt(4+i*5); --! # of sub header
    end generate;

    e_cnt_farm_events : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => events_to_farm_cnt, i_ena => farm_header, i_reset_n => i_reset_n_250, i_clk => i_clk_250 );


    --! data_generator_a10
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_gen_link : entity work.data_generator_a10
    generic map (
        DATA_TYPE => DATA_TYPE,
        go_to_sh => 3,
        test_error => false,
        go_to_trailer => 4--,
    )
    port map (
        enable_pix          => i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK),
        i_dma_half_full     => '0',
        random_seed         => (others => '1'),
        data_pix_generated  => gen_link,
        datak_pix_generated => gen_link_k,
        data_pix_ready      => open,
        start_global_time   => (others => '0'),
        delay               => (others => '0'),
        slow_down           => i_writeregs_156(DATAGENERATOR_DIVIDER_REGISTER_W),
        state_out           => open,

        i_reset_n           => i_resets_n_156(RESET_BIT_DATAGEN),
        i_clk               => i_clk_156--,
    );

    e_data_gen_error_test : entity work.data_generator_a10
    generic map (
        DATA_TYPE => DATA_TYPE,
        go_to_sh => 3,
        test_error => true,
        go_to_trailer => 4--,
    )
    port map (
        enable_pix          => i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK),
        i_dma_half_full     => '0',
        random_seed         => (others => '1'),
        data_pix_generated  => gen_link_error,
        datak_pix_generated => gen_link_k_error,
        data_pix_ready      => open,
        start_global_time   => (others => '0'),
        delay               => (others => '0'),
        slow_down           => i_writeregs_156(DATAGENERATOR_DIVIDER_REGISTER_W),
        state_out           => open,

        i_reset_n           => i_resets_n_156(RESET_BIT_DATAGEN),
        i_clk               => i_clk_156--,
    );

    gen_link_data : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE

        process(i_clk_156, i_reset_n_156)
        begin
        if ( i_reset_n_156 = '0' ) then
            rx(i)   <= (others => '0');
            rx_k(i) <= (others => '0');
        elsif rising_edge( i_clk_156 ) then
            if ( i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK) = '1' ) then
                if ( i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_TEST_ERROR) = '1' and i = 0 ) then
                    rx(i)   <= gen_link_error;
                    rx_k(i) <= gen_link_k_error;
                else
                    rx(i)   <= gen_link;
                    rx_k(i) <= gen_link_k;
                end if;
            else
                rx(i)   <= i_rx(i);
                rx_k(i) <= i_rx_k(i);
            end if;
        end if;
        end process;

    END GENERATE gen_link_data;


    --! generate link_to_fifo_32
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    gen_link_fifos : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE

        -- TODO: If its halffull than write only header (no hits) and write overflow into subheader
        --       If its full stop --> tell MIDAS --> stop run --> no event mixing
        -- TODO: different lookup for scifi
        e_link_to_fifo_32 : entity work.link_to_fifo_32
        generic map (
            g_LOOPUP_NAME        => g_LOOPUP_NAME,
            is_FARM              => false,
            SKIP_DOUBLE_SUB      => false,
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_rx            => rx(i),
            i_rx_k          => rx_k(i),
            i_linkid        => work.mudaq.link_36_to_std(i),

            o_q             => rx_q(i),
            i_ren           => rx_ren(i),
            o_rdempty       => rx_rdempty(i),

            o_counter(0)    => link_to_fifo_cnt(0+i*5),
            o_counter(1)    => link_to_fifo_cnt(1+i*5),
            o_counter(2)    => link_to_fifo_cnt(2+i*5),
            o_counter(3)    => link_to_fifo_cnt(3+i*5),
            o_counter(4)    => link_to_fifo_cnt(4+i*5),

            i_reset_n_156   => i_reset_n_156,
            i_clk_156       => i_clk_156,

            i_reset_n_250   => reset_250_n,
            i_clk_250       => i_clk_250--,
        );

        -- map outputs
        sop(i)      <= '1' when rx_q(i)(34 downto 32) = "010" else '0';
        shop(i)     <= '1' when rx_q(i)(34 downto 32) = "111" else '0';
        eop(i)      <= '1' when rx_q(i)(34 downto 32) = "001" else '0';
        hit(i)      <= '1' when rx_q(i)(34 downto 32) = "000" else '0';
        t0(i)       <= '1' when rx_q(i)(34 downto 32) = "100" else '0';
        t1(i)       <= '1' when rx_q(i)(34 downto 32) = "101" else '0';
        rx_q_s(i)   <= rx_q(i)(31 downto 0);

    END GENERATE gen_link_fifos;


    --! stream merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_stream : entity work.swb_stream_merger
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH,
        W => 32,
        N => g_NLINKS_DATA--,
    )
    port map (
        i_rdata     => rx_q_s,
        i_rsop      => sop,
        i_reop      => eop,
        i_rempty    => rx_rdempty,
        i_rmask_n   => i_rmask_n,
        o_rack      => stream_rack,

        -- farm data
        o_wdata     => stream_rdata,
        o_rempty    => stream_rempty,
        i_ren       => stream_ren,
        o_wsop      => stream_header,
        o_weop      => stream_trailer,

        -- data for debug readout
        o_wdata_debug   => stream_rdata_debug,
        o_rempty_debug  => stream_rempty_debug,
        i_ren_debug     => stream_ren_debug,
        o_wsop_debug    => stream_header_debug,
        o_weop_debug    => stream_trailer_debug,

        o_counters  => stream_counters,

        i_en        => i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM),
        i_reset_n   => i_resets_n_250(RESET_BIT_SWB_STREAM_MERGER),
        i_clk       => i_clk_250--,
    );


    --! time merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_time_merger : entity work.swb_time_merger
    generic map (
        g_ADDR_WIDTH    => g_ADDR_WIDTH,
        g_NLINKS_DATA   => g_NLINKS_DATA,
        DATA_TYPE       => DATA_TYPE--,
    )
    port map (
        i_rx            => rx_q_s,
        i_rsop          => sop,
        i_reop          => eop,
        i_rshop         => shop,
        i_hit           => hit,
        i_t0            => t0,
        i_t1            => t1,
        i_rempty        => rx_rdempty,
        i_rmask_n       => i_rmask_n,
        o_rack          => merger_rack,

        -- farm data
        o_wdata         => merger_rdata,
        o_rempty        => merger_rempty,
        i_ren           => merger_ren,
        o_wsop          => merger_header,
        o_weop          => merger_trailer,
        o_werp          => merger_error,

        -- data for debug readout
        o_wdata_debug   => merger_rdata_debug,
        o_rempty_debug  => merger_rempty_debug,
        i_ren_debug     => merger_ren_debug,
        o_wsop_debug    => merger_header_debug,
        o_weop_debug    => merger_trailer_debug,
        o_werp_debug    => merger_error_debug,

        o_error         => open,

        i_en            => i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER),
        i_reset_n       => i_resets_n_250(RESET_BIT_SWB_TIME_MERGER),
        i_clk           => i_clk_250--,
    );


    --! readout switches
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    rx_ren          <=  stream_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        (others => '0');

    builder_data    <=  stream_rdata_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rdata_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        (others => '0');
    builder_rempty  <=  stream_rempty_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rempty_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    builder_header  <=  stream_header_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_header_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    builder_trailer <=  stream_trailer_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_trailer_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    builder_error   <=  '0'                when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_error_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    stream_ren_debug <= builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else '0';
    merger_ren_debug <= builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else '0';

    farm_data       <=  stream_rdata when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rdata when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        (others => '0');
    farm_rempty     <=  stream_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    farm_header     <=  stream_header when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_header when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    farm_trailer    <=  stream_trailer when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_trailer when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    stream_ren      <=  farm_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else '0';
    merger_ren      <=  farm_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else '0';


    --! event builder used for the debug readout on the SWB
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_event_builder : entity work.swb_midas_event_builder
    generic map(
        DATA_TYPE           => DATA_TYPE--,
    )
    port map (
        i_rx                => builder_data,
        i_rempty            => builder_rempty,
        i_header            => builder_header,
        i_trailer           => builder_trailer,
        i_error             => builder_error,

        i_get_n_words       => i_writeregs_250(GET_N_DMA_WORDS_REGISTER_W),
        i_dmamemhalffull    => i_dmamemhalffull,
        i_wen               => i_writeregs_250(DMA_REGISTER_W)(DMA_BIT_ENABLE),

        o_data              => o_dma_data,
        o_wen               => o_dma_wren,
        o_ren               => builder_rack,
        o_endofevent        => o_endofevent,
        o_dma_cnt_words     => o_dma_cnt_words,
        o_done              => o_dma_done,

        o_counters          => builder_counters,

        i_reset_n_250       => reset_250_n,
        i_clk_250           => i_clk_250--,
    );


    --! generate farm output data
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    o_farm_data  <= x"000000BC" when farm_rempty  = '1' else farm_data;
    o_farm_datak <= "0001"      when farm_rempty  = '1' else
                    "0001"      when farm_header  = '1' else
                    "0001"      when farm_trailer = '1' else
                    "0000";
    farm_rack    <= not farm_rempty;

end architecture;
