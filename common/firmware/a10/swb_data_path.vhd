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
    g_NLINKS_TOTL : positive := 64;
    g_NLINKS_FARM : positive := 8;
    g_NLINKS_DATA : positive := 8;
    LINK_FIFO_ADDR_WIDTH : positive := 10;
    TREE_w : positive := 10;
    TREE_r : positive := 10;
    SWB_ID : std_logic_vector(7 downto 0) := x"01";
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE : std_logic_vector(7 downto 0) := x"01"--;
);
port(
    i_clk_156        : in  std_logic;
    i_clk_250        : in  std_logic;

    i_reset_n_156    : in  std_logic;
    i_reset_n_250    : in  std_logic;

    i_resets_n_156   : in  std_logic_vector(31 downto 0);
    i_resets_n_250   : in  std_logic_vector(31 downto 0);

    i_rx             : in  work.util.slv32_array_t(g_NLINKS_DATA-1 downto 0);
    i_rx_k           : in  work.util.slv4_array_t(g_NLINKS_DATA-1 downto 0);
    i_rmask_n        : in  std_logic_vector(g_NLINKS_DATA-1 downto 0);

    i_writeregs_156  : in  work.util.slv32_array_t(63 downto 0);
    i_writeregs_250  : in  work.util.slv32_array_t(63 downto 0);

    o_counter_156    : out work.util.slv32_array_t(g_NLINKS_DATA*5-1 downto 0);
    o_counter_250    : out work.util.slv32_array_t(4 downto 0);

    i_dmamemhalffull : in  std_logic;

    o_farm_data      : out work.util.slv32_array_t(g_NLINKS_FARM - 1  downto 0);
    o_farm_data_valid: out work.util.slv2_array_t(g_NLINKS_FARM - 1  downto 0);

    o_dma_wren       : out std_logic;
    o_dma_cnt_words  : out std_logic_vector (31 downto 0);
    o_dma_done       : out std_logic;
    o_endofevent     : out std_logic;
    o_dma_data       : out std_logic_vector (255 downto 0)--;
);
end entity;

architecture arch of swb_data_path is

    signal reset_250_n : std_logic;

    --! constant
    constant W : positive := g_NLINKS_FARM*32+g_NLINKS_FARM*6;

    --! data gen links
    signal gen_link : std_logic_vector(31 downto 0);
    signal gen_link_k : std_logic_vector(3 downto 0);
    signal gen_data, gen_q : std_logic_vector(W-1 downto 0);
    signal gen_rempty, gen_re, gen_we, gen_full : std_logic;

    --! data link signals
    signal rx : work.util.slv32_array_t(g_NLINKS_DATA-1 downto 0);
    signal rx_k : work.util.slv4_array_t(g_NLINKS_DATA-1 downto 0);
    signal rx_ren, rx_mask_n, rx_rdempty : std_logic_vector(g_NLINKS_DATA-1 downto 0) := (others => '0');
    signal rx_q : work.util.slv34_array_t(g_NLINKS_DATA-1 downto 0) := (others => (others => '0'));
    signal sop, eop, shop : std_logic_vector(g_NLINKS_DATA-1 downto 0) := (others => '0');

    --! stream merger
    signal stream_rdata : std_logic_vector(31 downto 0);
    signal stream_counters : work.util.slv32_array_t(0 downto 0);
    signal stream_rempty, stream_ren, stream_header, stream_trailer : std_logic;
    signal stream_rack : std_logic_vector(g_NLINKS_DATA-1 downto 0);

    --! timer merger
    signal merger_rdata : std_logic_vector(W-1 downto 0);
    signal merger_rdata_debug : std_logic_vector(31 downto 0);
    signal merger_rempty, merger_rempty_debug, merger_ren, merger_header, merger_trailer, merger_error : std_logic;
    signal merger_rack : std_logic_vector (g_NLINKS_DATA-1 downto 0);

    --! event builder
    signal builder_data : std_logic_vector(31 downto 0);
    signal builder_counters : work.util.slv32_array_t(3 downto 0);
    signal builder_rempty, builder_rack, builder_header, builder_trailer : std_logic;

    --! links to farm
    signal merged_farm_data : std_logic_vector (g_NLINKS_FARM * 32 - 1  downto 0);
    signal merged_farm_data_valid : std_logic_vector (g_NLINKS_FARM * 2 - 1  downto 0);
    signal farm_data : std_logic_vector(W-1 downto 0);
    signal farm_rack, farm_rempty, all_padding : std_logic;

    --! status counters
    signal link_to_fifo_cnt : work.util.slv32_array_t((g_NLINKS_DATA*5)-1 downto 0);

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

    -- 156 MHz counters
    generate_rdata : for i in 0 to g_NLINKS_DATA - 1 generate
        o_counter_156(0+i*5) <= link_to_fifo_cnt(0+i*5); --! fifo almost_full
        o_counter_156(1+i*5) <= link_to_fifo_cnt(1+i*5); --! fifo wrfull
        o_counter_156(2+i*5) <= link_to_fifo_cnt(2+i*5); --! # of skip event
        o_counter_156(3+i*5) <= link_to_fifo_cnt(3+i*5); --! # of events
        o_counter_156(4+i*5) <= link_to_fifo_cnt(4+i*5); --! # of sub header
    end generate;


    --! data_generator_a10
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_gen_link : entity work.data_generator_a10
    generic map (
        DATA_TYPE => DATA_TYPE,
        go_to_sh => 3,
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

    gen_link_data : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE

        process(i_clk_156, i_reset_n_156)
        begin
        if ( i_reset_n_156 = '0' ) then
            rx(i)   <= (others => '0');
            rx_k(i) <= (others => '0');
        elsif rising_edge( i_clk_156 ) then
            if (i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK) = '1') then
                rx(i)   <= gen_link;
                rx_k(i) <= gen_link_k;
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
        e_link_to_fifo_32 : entity work.link_to_fifo_32
        generic map (
            SKIP_DOUBLE_SUB      => 2, -- 1 means skip 2 means dont skip
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_rx            => rx(i),
            i_rx_k          => rx_k(i),

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

        sop(i)  <= '1' when rx_q(i)(33 downto 32) = "10" else '0';
        shop(i) <= '1' when rx_q(i)(33 downto 32) = "11" else '0';
        eop(i)  <= '1' when rx_q(i)(33 downto 32) = "01" else '0';

    END GENERATE gen_link_fifos;


    --! stream merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_stream : entity work.swb_stream_merger
    generic map (
        W => 34,
        N => g_NLINKS_DATA--,
    )
    port map (
        i_rdata     => rx_q,
        i_rsop      => sop,
        i_reop      => eop,
        i_rempty    => rx_rdempty,
        i_rmask_n   => i_rmask_n,
        i_en        => i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM),
        o_rack      => stream_rack,

        o_wdata     => stream_rdata,
        o_rempty    => stream_rempty,
        i_ren       => stream_ren,
        o_wsop      => stream_header,
        o_weop      => stream_trailer,

        o_counters  => stream_counters,

        i_reset_n   => reset_250_n,
        i_clk       => i_clk_250--,
    );


    --! time merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_time_merger : entity work.swb_time_merger
    generic map (
        W               => W,
        TREE_w          => TREE_w,
        TREE_r          => TREE_r,
        DATA_TYPE       => DATA_TYPE,
        g_NLINKS_DATA   => g_NLINKS_DATA,
        g_NLINKS_FARM   => g_NLINKS_FARM--,
    )
    port map (
        i_rx            => rx_q,
        i_rsop          => sop,
        i_reop          => eop,
        i_rshop         => shop,
        i_rempty        => rx_rdempty,
        i_rmask_n       => i_rmask_n,
        o_rack          => merger_rack,

        -- output stream
        o_q             => merger_rdata,
        i_debug         => i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER),
        o_q_debug       => merger_rdata_debug,
        o_rempty        => merger_rempty,
        o_rempty_debug  => merger_rempty_debug,
        i_ren           => merger_ren,
        o_header_debug  => merger_header,
        o_trailer_debug => merger_trailer,
        o_error         => open,

        i_reset_n       => reset_250_n,
        i_clk           => i_clk_250--,
    );


    --! readout switches
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    builder_data  <=  stream_rdata when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                      merger_rdata_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                      (others => '0');
    builder_rempty  <=  stream_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rempty_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    builder_header  <=  stream_header when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_header when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    builder_trailer <=  stream_trailer when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_trailer when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    stream_ren <= builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else '0';
    merger_ren <= farm_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_FARM) = '1' else 
                  builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else 
                  '0';
    rx_ren <=   stream_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                merger_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                (others => '0');
    farm_data <=    merger_rdata when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                    gen_q when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_MERGER) = '1' else
                    (others => '0');
    farm_rempty <=  merger_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                    gen_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_MERGER) = '1' else
                    '0';
    gen_re <= farm_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_MERGER) = '1' else '0';


    --! event builder
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


    --! data_generator_merged_data
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_gen_merged : entity work.data_generator_merged_data
    port map(
        i_clk       => i_clk_250,
        i_reset_n   => reset_250_n,
        i_en        => not gen_full,
        i_sd        => x"00000002",
        o_data      => gen_data,
        o_data_we   => gen_we,
        o_state     => open--,
    );

--    e_merger_fifo : entity work.ip_scfifo
--    generic map (
--        ADDR_WIDTH      => 10,
--        DATA_WIDTH      => W--,
--    )
--    port map (
--        data            => gen_data,
--        wrreq           => gen_we,
--        rdreq           => gen_re,
--        clock           => i_clk_250,
--        q               => gen_q,
--        full            => gen_full,
--        empty           => gen_rempty,
--        sclr            => not reset_250_n--,
--    );


    --! swb_data_merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_merger : entity work.swb_data_merger
    generic map (
        NLINKS      => g_NLINKS_FARM,
        SWB_ID      => SWB_ID,
        DATA_TYPE   => DATA_TYPE--;
    )
    port map (
        i_reset_n   => reset_250_n,
        i_clk       => i_clk_250,

        i_data      => farm_data,
        i_empty     => farm_rempty,

        o_ren       => farm_rack,

        o_data      => merged_farm_data,
        o_data_valid=> merged_farm_data_valid--,
    );

    all_padding <= '1' when work.util.and_reduce(merged_farm_data(227 downto 0)) = '1' else '0';

    gen_farm_out : FOR i in 0 to g_NLINKS_FARM - 1 GENERATE
        o_farm_data(i)          <= merged_farm_data(32 * i + 31 downto 32 * i);
        o_farm_data_valid(i)    <= "00" when all_padding = '1' else merged_farm_data_valid(2 * i + 1 downto 2 * i);
    END GENERATE gen_farm_out;

end architecture;
