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


entity swb_data_path is
generic (
    g_NLINKS_TOTL : integer := 64;
    g_NLINKS_FARM : integer := 8;
    g_NLINKS_DATA : integer := 8;
    LINK_FIFO_ADDR_WIDTH : integer := 10;
    TREE_w : integer := 10;
    TREE_r : integer := 10;
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
    
    i_rx             : in  work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    i_rx_k           : in  work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    i_rmask_n        : in  std_logic_vector(g_NLINKS_TOTL-1 downto 0);

    i_writeregs_156     : in    work.util.slv32_array_t(63 downto 0);
    i_writeregs_250     : in    work.util.slv32_array_t(63 downto 0);

    o_counter        : out work.util.slv32_array_t(5+(g_NLINKS_TOTL*3)-1 downto 0);

    i_dmamemhalffull : in  std_logic;
    
    o_farm_data      : out std_logic_vector (g_NLINKS_FARM * 32 - 1  downto 0);
    o_farm_datak     : out std_logic_vector (g_NLINKS_FARM * 4 - 1  downto 0);
    o_fram_wen       : out std_logic;

    o_dma_wren       : out std_logic;
    o_dma_done       : out std_logic;
    o_endofevent     : out std_logic; 
    o_dma_data       : out std_logic_vector (255 downto 0)--;
);
end entity;

architecture arch of swb_data_path is


    --! data gen links
    signal gen_link : std_logic_vector(31 downto 0);
    signal gen_link_k : std_logic_vector(3 downto 0);

    --! data link signals
    signal rx : work.util.slv32_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_k : work.util.slv4_array_t(g_NLINKS_TOTL-1 downto 0);
    signal rx_mask_n, rx_rdempty, rx_ren : std_logic_vector (g_NLINKS_TOTL - 1 downto 0);
    signal rx_q : work.util.slv38_array_t(g_NLINKS_TOTL - 1 downto 0);
    signal sop, eop, shop : std_logic_vector(g_NLINKS_TOTL-1 downto 0);

    --! stream merger
    signal stream_rdata : std_logic_vector(31 downto 0);
    signal stream_counters : work.util.slv32_array_t(0 downto 0);
    signal stream_rempty, stream_ren, stream_header, stream_trailer : std_logic;
    signal stream_rack : std_logic_vector(g_NLINKS_TOTL-1 downto 0);

    --! timer merger
    signal merger_header, merger_trailer, merger_error : std_logic;

    --! event builder
    signal builder_counters : work.util.slv32_array_t(3 downto 0);


    --! status counters
    signal link_to_fifo_cnt : work.util.slv32_array_t((g_NLINKS_TOTL*3)-1 downto 0);

begin


    --! status counter
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! TODO: add this to counters
    -- tag_fifo_empty;
    -- dma_write_state;
    -- rx_rdempty;
    o_counter(0) <= stream_counters(0);  --! e_stream_fifo full
    o_counter(1) <= builder_counters(0); --! bank_builder_idle_not_header
    o_counter(2) <= builder_counters(1); --! bank_builder_skip_event_dma
    o_counter(3) <= builder_counters(2); --! bank_builder_ram_full
    o_counter(4) <= builder_counters(3); --! bank_builder_tag_fifo_full
    generate_rdata : for i in 0 to g_NLINKS_TOTL - 1 generate
        o_counter(5+i*3) <= link_to_fifo_cnt(0+i*3); --! fifo almost_full
        o_counter(6+i*3) <= link_to_fifo_cnt(1+i*3); --! fifo wrfull
        o_counter(7+i*3) <= link_to_fifo_cnt(2+i*3); --! # of skip event
    end generate;


    --! data_generator_a10
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_gen_link : entity work.data_generator_a10
    generic map (
        go_to_trailer => 4,
        go_to_sh => 3--,
    )
    port map (
        reset               => not i_resets_n_156(RESET_BIT_DATAGEN),
        enable_pix          => i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_GEN_LINK),
        i_dma_half_full     => i_dmamemhalffull,
        random_seed         => (others => '1'),
        data_pix_generated  => gen_link,
        datak_pix_generated => gen_link_k,
        data_pix_ready      => open,
        start_global_time   => (others => '0'),
        slow_down           => i_writeregs_156(DATAGENERATOR_DIVIDER_REGISTER_W),
        state_out           => open,
        clk                 => i_clk_156--,
    );

    gen_link_data : FOR i in 0 to g_NLINKS_TOTL - 1 GENERATE
        process(i_clk_156, i_reset_n_156)
        begin
        if ( i_reset_n_156 = '0' ) then
            rx(i)   <= (others => '0');
            rx_k(i) <= (others => '0');
        elsif rising_edge(i_clk_156) then
            if (i_writeregs_156(SWB_READOUT_STATE_REGISTER_W)(USE_GEN_LINK) = '1') then
                rx(i)   <= gen_link;
                rx_k(i) <= gen_link_k;
            elsif ( i < g_NLINKS_DATA ) then
                rx(i)   <= i_rx(i);
                rx_k(i) <= i_rx_k(i);
            else
                rx(i)   <= (others => '0');
                rx_k(i) <= (others => '0');
            end if;
        end if;
        end process;
    END GENERATE gen_link_data;

    --! generate link_to_fifo_32
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    gen_link_fifos : FOR i in 0 to g_NLINKS_TOTL - 1 GENERATE
        
        e_link_to_fifo_32 : entity work.link_to_fifo_32
        generic map (
            LINK_FIFO_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH--;
        )
        port map (
            i_rx            => rx(i),
            i_rx_k          => rx_k(i),
            
            o_q             => rx_q(i),
            i_ren           => rx_ren(i),
            o_rdempty       => rx_rdempty(i),

            o_counter(0)  => link_to_fifo_cnt(0+i*3),
            o_counter(1)  => link_to_fifo_cnt(1+i*3),
            o_counter(2)  => link_to_fifo_cnt(2+i*3),

            i_reset_n_156   => i_reset_n_156,
            i_clk_156       => i_clk_156,

            i_reset_n_250   => i_reset_n_250,
            i_clk_250       => i_clk_250--;
        );
  
        sop(i) <= rx_q(i)(36);
        shop(i) <= '1' when rx_q(i)(37 downto 36) = "00" and rx_q(i)(31 downto 26) = "111111" else '0';
        eop(i) <= rx_q(i)(37);

    END GENERATE gen_link_fifos;


    --! stream merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_stream : entity work.swb_stream_merger
    generic map (
        W => 38,
        N => g_NLINKS_TOTL--,
    )
    port map (
        i_rx        => rx_q,
        i_rsop      => sop,
        i_reop      => eop,
        i_rempty    => rx_rdempty,
        i_rmask_n   => i_rmask_n,
        o_rack      => stream_rack,

        o_q         => stream_rdata,
        o_rempty    => stream_rempty,
        i_ren       => stream_ren,
        o_header    => stream_header,
        o_trailer   => stream_trailer,

        o_counters  => stream_counters,

        i_reset_n   => i_reset_n_250,
        i_clk       => i_clk_250--,
    );


    --! time merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_time_merger : entity work.swb_time_merger
    generic map (
        W           => 8*32+8*6,
        TREE_w      => TREE_w,
        TREE_r      => TREE_r,
        DATA_TYPE   => DATA_TYPE,
        g_NLINKS    => g_NLINKS_TOTL--,
    )
    port map (
        i_rx        => rx_q,
        i_rsop      => sop,
        i_reop      => eop,
        i_rshop     => shop,
        i_rempty    => rx_rdempty,
        i_rmask_n   => i_rmask_n,
        o_rack      => merger_rack,

        -- output strem
        o_q         => merger_rdata,
        o_q_debug   => merger_rdata_debug,
        o_rempty    => merger_rempty,
        i_ren       => merger_ren,
        o_header    => merger_header,
        o_trailer   => merger_trailer,
        o_error     => open,

        i_reset_n   => i_reset_n_250,
        i_clk       => i_clk_250--,
    );


    --! readout switches
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    link_idx <= to_integer(unsigned(i_writeregs_250(SWB_READOUT_LINK_REGISTER_W)));
    rx_builder  <=  stream_rdata when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_STREAM) = '1' else
                    merger_rdata_debug when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                    rx_q(link_idx) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_LINK) = '1' else
                    (others => '0');
    rempty_builder  <=  stream_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_STREAM) = '1' else
                        merger_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                        rx_rdempty(link_idx) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_LINK) = '1' else
                        '0';
    header_builder  <=  stream_header when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_STREAM) = '1' else
                        merger_header when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                        sop(link_idx) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_LINK) = '1' else
                        '0';
    trailer_builder <=  stream_trailer when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_STREAM) = '1' else
                        merger_trailer when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                        eop(link_idx) when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_LINK) = '1' else
                        '0';
    stream_ren <= builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_STREAM) = '1' else '0';
    merger_ren <= builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else 
                  farm_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_FARM) = '1' else 
                  '0';
    rx_ren(link_idx) <= builder_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_LINK) = '1' else
                        '0';
    rx_ren <=   stream_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_STREAM) = '1' else
                merger_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                (others => '0');
    farm_data <=    merger_rdata when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                    gen_q when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_GEN_MERGER) = '1' else
                    (others => '0');
    farm_rempty <=  merger_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_MERGER) = '1' else
                    gen_rempty when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_GEN_MERGER) = '1' else
                    '0';
    gen_re <= farm_rack when i_writeregs_250(SWB_READOUT_STATE_REGISTER_W)(USE_GEN_MERGER) = '1' else '0';


    --! event builder
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_event_builder : entity work.swb_midas_event_builder
    port map (
        i_rx                => rx_builder,
        i_rempty            => rempty_builder,
        i_header            => header_builder,
        i_trailer           => trailer_builder,
        
        i_get_n_words       => i_writeregs_250(GET_N_DMA_WORDS_REGISTER_W),
        i_dmamemhalffull    => i_dmamemhalffull,
        i_wen               => i_writeregs_250(DMA_REGISTER_W)(DMA_BIT_ENABLE),

        o_data              => o_dma_data,
        o_wen               => o_dma_wren,
        o_ren               => builder_rack,
        o_endofevent        => o_endofevent,
        o_done              => o_dma_done,

        o_counters          => builder_counters,

        i_reset_n_250       => i_reset_n_250,
        i_clk_250           => i_clk_250--,
    );


    --! data_generator_merged_data
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_gen_merged : entity work.data_generator_merged_data
    port map(
        i_clk       => i_clk_250,
        i_reset_n   => i_reset_n_250,
        i_en        => not gen_full,
        i_sd        => x"00000002",
        o_data      => gen_data,
        o_data_we   => gen_we,
        o_state     => open--,
    );

    e_merger_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 10,
        DATA_WIDTH      => g_NLINKS_FARM * 38,
        DEVICE          => "Arria 10"--,
    )
    port map (
        data            => gen_data,
        wrreq           => gen_we,
        rdreq           => gen_re,
        clock           => i_clk_250,
        q               => gen_q,
        full            => gen_full,
        empty           => gen_empty,
        sclr            => not i_reset_n_250--,
    );


    --! swb_data_merger
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_data_merger : entity work.swb_data_merger
    generic map (
        NLINKS => g_NLINKS_FARM,
        SWB_ID => SWB_ID,
        DT     => DATA_TYPE--;
    )
    port map (
        i_reset_n   => i_reset_n_250,
        i_clk       => i_clk_250,

        i_data      => farm_data,
        i_empty     => farm_rempty,

        o_ren       => farm_rack,
        o_wen       => o_fram_wen,

        o_data      => o_farm_data,
        o_datak     => o_farm_datak--,
    );

end architecture;
