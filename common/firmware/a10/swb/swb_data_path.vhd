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
    -- Data type: "00" = pixel, "01" = scifi, "10" = tiles
    DATA_TYPE : std_logic_vector(1 downto 0) := "00"--;
);
port(
    -- clk and reset signals
    i_clk            : in  std_logic;
    i_reset_n        : in  std_logic;
    i_resets_n       : in  std_logic_vector(31 downto 0);
    
    -- link inputs
    i_rx             : in  work.mu3e.link_array_t(g_NLINKS_DATA-1 downto 0);
    i_rmask_n        : in  std_logic_vector(g_NLINKS_DATA-1 downto 0);

    -- pcie regs
    i_writeregs      : in  work.util.slv32_array_t(63 downto 0);

    -- counters
    o_counter        : out work.util.slv32_array_t(g_NLINKS_DATA*5+6-1 downto 0);

    -- farm data
    o_farm_data      : out work.mu3e.link_t;

    -- dma
    i_dmamemhalffull : in  std_logic;
    o_dma_wren       : out std_logic;
    o_dma_cnt_words  : out std_logic_vector (31 downto 0);
    o_dma_done       : out std_logic;
    o_endofevent     : out std_logic;
    o_dma_data       : out std_logic_vector (255 downto 0)--;
);
end entity;

architecture arch of swb_data_path is

    signal reset_250_n : std_logic;

    --! data gen links
    signal gen_link, gen_link_error : work.mu3e.link_t;

    --! data link signals
    signal rx : work.mu3e.link_array_t(g_NLINKS_DATA-1 downto 0);
    signal rx_ren, rx_mask_n, rx_rdempty : std_logic_vector(g_NLINKS_DATA-1 downto 0) := (others => '0');
    signal rx_q : work.mu3e.link_array_t(g_NLINKS_DATA-1 downto 0);

    --! stream merger
    signal stream_rdata, stream_rdata_debug : work.mu3e.link_t;
    signal stream_counters : work.util.slv32_array_t(0 downto 0);
    signal stream_rempty, stream_ren : std_logic;
    signal stream_rempty_debug, stream_ren_debug : std_logic;
    signal stream_rack : std_logic_vector(g_NLINKS_DATA-1 downto 0);

    --! timer merger
    signal merger_rdata : work.mu3e.link_t;
    signal merger_rdata_debug : work.mu3e.link_t;
    signal merger_rempty, merger_ren, merger_header, merger_trailer, merger_error : std_logic;
    signal merger_rempty_debug, merger_ren_debug : std_logic;
    signal merger_rack : std_logic_vector (g_NLINKS_DATA-1 downto 0);

    --! event builder
    signal builder_data : work.mu3e.link_t;
    signal builder_counters : work.util.slv32_array_t(3 downto 0);
    signal builder_rempty, builder_rack : std_logic;

    --! links to farm
    signal farm_data : work.mu3e.link_t;
    signal farm_rack, farm_rempty : std_logic;

    --! status counters
    signal link_to_fifo_cnt : work.util.slv32_array_t((g_NLINKS_DATA*5)-1 downto 0);
    signal events_to_farm_cnt : std_logic_vector(31 downto 0);

begin

    --! status counter
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! TODO: add this to counters
    -- tag_fifo_empty;
    -- dma_write_state;
    -- rx_rdempty;

    -- dma and farm counters
    o_counter(0) <= stream_counters(0);  --! e_stream_fifo full
    o_counter(1) <= builder_counters(0); --! bank_builder_idle_not_header
    o_counter(2) <= builder_counters(1); --! bank_builder_skip_event_dma
    o_counter(3) <= builder_counters(2); --! bank_builder_ram_full
    o_counter(4) <= builder_counters(3); --! bank_builder_tag_fifo_full
    o_counter(5) <= events_to_farm_cnt;  --! events send to the farm
    
    -- link to fifo counters
    generate_rdata : for i in 0 to g_NLINKS_DATA - 1 generate
        o_counter(6+0+i*5) <= link_to_fifo_cnt(0+i*5); --! fifo almost_full
        o_counter(6+1+i*5) <= link_to_fifo_cnt(1+i*5); --! fifo wrfull
        o_counter(6+2+i*5) <= link_to_fifo_cnt(2+i*5); --! # of skip event
        o_counter(6+3+i*5) <= link_to_fifo_cnt(3+i*5); --! # of events
        o_counter(6+4+i*5) <= link_to_fifo_cnt(4+i*5); --! # of sub header
    end generate;
    
    e_cnt_farm_events : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => events_to_farm_cnt, i_ena => farm_data.sop, i_reset_n => i_reset_n, i_clk => i_clk );


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
        i_enable            => i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK),
        i_dma_half_full     => '0',
        i_seed              => (others => '1'),
        o_data              => gen_link,
        i_start_global_time => (others => '0'),
        i_delay             => (others => '0'),
        i_slow_down         => i_writeregs(DATAGENERATOR_DIVIDER_REGISTER_W),
        o_state             => open,
        
        i_reset_n           => i_resets_n(RESET_BIT_DATAGEN),
        i_clk               => i_clk--,
    );
    
    e_data_gen_error_test : entity work.data_generator_a10
    generic map (
        DATA_TYPE => DATA_TYPE,
        go_to_sh => 3,
        test_error => true,
        go_to_trailer => 4--,
    )
    port map (
        i_enable            => i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK),
        i_dma_half_full     => '0',
        i_seed              => (others => '1'),
        o_data              => gen_link_error,
        i_start_global_time => (others => '0'),
        i_delay             => (others => '0'),
        i_slow_down         => i_writeregs(DATAGENERATOR_DIVIDER_REGISTER_W),
        o_state             => open,
        
        i_reset_n           => i_resets_n(RESET_BIT_DATAGEN),
        i_clk               => i_clk--,
    );
    
    gen_link_data : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE

        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n = '0' ) then
            rx(i)   <= work.mu3e.LINK_ZERO;
        elsif ( rising_edge(i_clk) ) then
            if ( i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_GEN_LINK) = '1' ) then
                if ( i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_TEST_ERROR) = '1' and i = 0 ) then
                    rx(i) <= work.mu3e.to_link(gen_link_error.data, gen_link_error.datak);
                else
                    rx(i) <= work.mu3e.to_link(gen_link.data, gen_link.datak);
                end if;
            else
                rx(i) <= i_rx(i);
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
        e_link_to_fifo_32 : entity work.link_to_fifo
        generic map (
            g_LOOPUP_NAME        => g_LOOPUP_NAME,
            is_FARM              => false,
            SKIP_DOUBLE_SUB      => false,
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--,
        )
        port map (
            i_rx            => rx(i),
            i_linkid        => work.mudaq.link_36_to_std(i),

            o_q             => rx_q(i),
            i_ren           => rx_ren(i),
            o_rdempty       => rx_rdempty(i),

            o_counter(0)    => link_to_fifo_cnt(0+i*5),
            o_counter(1)    => link_to_fifo_cnt(1+i*5),
            o_counter(2)    => link_to_fifo_cnt(2+i*5),
            o_counter(3)    => link_to_fifo_cnt(3+i*5),
            o_counter(4)    => link_to_fifo_cnt(4+i*5),

            i_reset_n       => i_reset_n,
            i_clk           => i_clk--,

        );

    END GENERATE gen_link_fifos;


    --! stream merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_stream : entity work.swb_stream_merger
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH,
        N => g_NLINKS_DATA--,
    )
    port map (
        i_rdata     => rx_q,
        i_rempty    => rx_rdempty,
        i_rmask_n   => i_rmask_n,
        o_rack      => stream_rack,

        -- farm data
        o_wdata     => stream_rdata,
        o_rempty    => stream_rempty,
        i_ren       => stream_ren,

        -- data for debug readout
        o_wdata_debug   => stream_rdata_debug,
        o_rempty_debug  => stream_rempty_debug,
        i_ren_debug     => stream_ren_debug,

        o_counters  => stream_counters,

        i_en        => i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM),
        i_reset_n   => i_resets_n(RESET_BIT_SWB_STREAM_MERGER),
        i_clk       => i_clk--,
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
        i_rx            => rx_q,
        i_rempty        => rx_rdempty,
        i_rmask_n       => i_rmask_n,
        o_rack          => merger_rack,

        -- farm data
        o_wdata         => merger_rdata,
        o_rempty        => merger_rempty,
        i_ren           => merger_ren,

        -- data for debug readout
        o_wdata_debug   => merger_rdata_debug,
        o_rempty_debug  => merger_rempty_debug,
        i_ren_debug     => merger_ren_debug,

        o_error         => open,

        i_en            => i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER),
        i_reset_n       => i_resets_n(RESET_BIT_SWB_TIME_MERGER),
        i_clk           => i_clk--,
    );


    --! readout switches
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    rx_ren          <=  stream_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        (others => '0');

    builder_data    <=  stream_rdata_debug when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rdata_debug when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        work.mu3e.LINK_ZERO;
    builder_rempty  <=  stream_rempty_debug when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rempty_debug when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    stream_ren_debug <= builder_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else '0';
    merger_ren_debug <= builder_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else '0';

    farm_data       <=  stream_rdata when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rdata when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        work.mu3e.LINK_ZERO;
    farm_rempty     <=  stream_rempty when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else
                        merger_rempty when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else
                        '0';
    stream_ren      <=  farm_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_STREAM) = '1' else '0';
    merger_ren      <=  farm_rack when i_writeregs(SWB_READOUT_STATE_REGISTER_W)(USE_BIT_MERGER) = '1' else '0';


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
        i_data_type         => DATA_TYPE,

        o_data              => o_dma_data,
        o_wen               => o_dma_wren,
        o_ren               => builder_rack,
        o_endofevent        => o_endofevent,
        o_dma_cnt_words     => o_dma_cnt_words,
        o_done              => o_dma_done,

        o_counters          => builder_counters,

        i_reset_n_250       => i_reset_n,
        i_clk_250           => i_clk--,
    );


    --! generate farm output data
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    o_farm_data  <= farm_data;
    farm_rack    <= not farm_rempty;

end architecture;
