--
-- Marius Koeppel, November 2020
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;

entity midas_event_builder is
generic (
    W : integer := 66;
    NLINKS : integer := 4;
    LINK_FIFO_ADDR_WIDTH : integer := 10;
    TREE_w : integer := 10;
    TREE_r : integer := 10;
    USE_ALIGNMENT : integer := 0--;
);
port (
    i_clk_data                  : in    std_logic;
    i_clk_dma                   : in    std_logic;

    i_reset_data_n              : in    std_logic;
    i_reset_dma_n               : in    std_logic;

    i_link_data                 : in    std_logic_vector (NLINKS * 32 - 1 downto 0);
    i_link_datak                : in    std_logic_vector (NLINKS * 4 - 1 downto 0);
    i_link_mask_n               : in    std_logic_vector (NLINKS - 1 downto 0);

    i_wen_reg                   : in    std_logic;
    i_get_n_words               : in    std_logic_vector (31 downto 0);
    i_dmamemhalffull            : in    std_logic;

    o_event_wren                : out   std_logic;
    o_endofevent                : out   std_logic;
    o_event_data                : out   std_logic_vector (255 downto 0);

    -- error / state signals
    o_state_out                 : out   std_logic_vector(3 downto 0);
    o_fifos_full                : out   std_logic_vector (NLINKS downto 0); -- fifos and dmamemhalffull
    o_all_done                  : out   std_logic_vector (NLINKS downto 0);
    o_done                      : out   std_logic;
    o_fifo_almost_full          : out   std_logic_vector(NLINKS - 1 downto 0);
    o_cnt_link_fifo_almost_full : out   std_logic_vector(31 downto 0);
    o_cnt_tag_fifo_full         : out   std_logic_vector(31 downto 0);
    o_cnt_ram_full              : out   std_logic_vector(31 downto 0);
    o_cnt_stream_fifo_full      : out   std_logic_vector(31 downto 0);
    o_cnt_dma_halffull          : out   std_logic_vector(31 downto 0);
    o_cnt_dc_link_fifo_full     : out   std_logic_vector(31 downto 0);
    o_cnt_skip_link_data        : out   std_logic_vector(31 downto 0);
    o_cnt_skip_event_dma        : out   std_logic_vector(31 downto 0);
    o_cnt_idle_not_header       : out   std_logic_vector(31 downto 0)--;
);
end entity;

architecture rtl of midas_event_builder is

    ----------------signals---------------------
    signal reset_data, reset_dma : std_logic;

    -- writerregs
    signal wen_reg : std_logic;
    signal get_n_words : std_logic_vector (31 downto 0);
    signal link_mask_n : std_logic_vector (NLINKS - 1 downto 0);

    -- link fifos
    signal link_fifo_wren, link_fifo_ren, link_fifo_ren_reg, link_fifo_empty, link_fifo_empty_reg, link_fifo_full, link_fifo_almost_full : std_logic_vector(NLINKS - 1 downto 0);
    signal link_data_f, link_dataq_f, link_dataq_f_reg : work.util.slv38_array_t(NLINKS - 1 downto 0);
    signal link_fifo_usedw, link_fifo_usedw_reg : std_logic_vector(LINK_FIFO_ADDR_WIDTH * NLINKS - 1 downto 0);
    signal sync_fifo_empty : std_logic_vector(NLINKS - 1 downto 0);
    signal sync_fifo_i_wrreq : std_logic_vector(NLINKS - 1 downto 0);
    type sync_fifo_t is array (NLINKS - 1 downto 0) of std_logic_vector(35 downto 0);
    signal sync_fifo_q : sync_fifo_t;
    signal sync_fifo_data : sync_fifo_t;

    -- event ram
    signal w_ram_data : std_logic_vector(31 downto 0);
    signal w_ram_add  : std_logic_vector(11 downto 0);
    signal w_ram_en   : std_logic;
    signal r_ram_data : std_logic_vector(255 downto 0);
    signal r_ram_add  : std_logic_vector(8 downto 0);

    -- tagging fifo
    type event_tagging_state_type is (
        event_head, event_num, event_tmp, event_size, bank_size, bank_flags, bank_name, bank_type, bank_length, bank_data, bank_set_length, trailer_name, trailer_type, trailer_length, trailer_data, trailer_set_length, event_set_size, bank_set_size, write_tagging_fifo, set_algin_word,
        EVENT_IDLE--,
    );

    signal event_tagging_state : event_tagging_state_type;
    signal data_flag 				: std_logic;
    signal cur_size_add 			: std_logic_vector(11 downto 0);
    signal cur_bank_size_add 	: std_logic_vector(11 downto 0);
    signal cur_bank_length_add : std_logic_vector(12 - 1 downto 0);

    signal w_ram_add_reg 	: std_logic_vector(11 downto 0);
    signal w_fifo_data      : std_logic_vector(11 downto 0);
    signal w_fifo_en        : std_logic;
    signal r_fifo_data      : std_logic_vector(11 downto 0);
    signal r_fifo_en        : std_logic;
    signal tag_fifo_empty   : std_logic;
    signal tag_fifo_full    : std_logic;
    signal last_event_add 	: std_logic_vector(11 downto 0);
    signal align_event_size : std_logic_vector(11 downto 0);

    -- midas event
    signal event_id 		: std_logic_vector(15 downto 0);
    signal trigger_mask 	: std_logic_vector(15 downto 0);
    signal serial_number : std_logic_vector(31 downto 0);
    signal time_tmp 		: std_logic_vector(31 downto 0);
    signal type_bank 		: std_logic_vector(31 downto 0);
    signal flags 			: std_logic_vector(31 downto 0);
    signal bank_size_cnt : std_logic_vector(31 downto 0);
    signal event_size_cnt: std_logic_vector(31 downto 0);

    -- event readout state machine
    type event_counter_state_type is (waiting, get_data, runing, skip_event);
    signal event_counter_state : event_counter_state_type;
    signal event_last_ram_add : std_logic_vector(8 downto 0);
    signal word_counter : std_logic_vector(31 downto 0);

    -- current link data/datak/empty
    signal sop, sop_reg, eop, eop_reg, shop, shop_reg : std_logic_vector(NLINKS-1 downto 0);
    signal stream_in_rempty : std_logic_vector(NLINKS-1 downto 0);
    signal stream_wdata, stream_rdata : std_logic_vector(35 downto 0);
    signal time_merger_hit : std_logic_vector(37 downto 0);
    signal stream_rempty, time_rempty, stream_rack, stream_wfull, stream_we : std_logic;
    signal link_data : std_logic_vector(31 downto 0);
    signal link_datak : std_logic_vector(3 downto 0);
    signal link_number : std_logic_vector(5 downto 0);
    signal link_header, link_trailer, link_error : std_logic;

    -- error cnt
    constant all_zero : std_logic_vector(NLINKS - 1 downto 0) := (others => '0');
    signal fifos_full, fifos_full_reg : std_logic_vector(NLINKS - 1 downto 0);
    signal cnt_link_fifo_almost_full : std_logic_vector(31 downto 0);
    signal cnt_tag_fifo_full : std_logic_vector(31 downto 0);
    signal cnt_ram_full : std_logic_vector(31 downto 0);
    signal cnt_stream_fifo_full : std_logic_vector(31 downto 0);
    signal cnt_dma_halffull : std_logic_vector(31 downto 0);
    signal cnt_dc_link_fifo_full : std_logic_vector(31 downto 0);
    signal cnt_skip_event_dma : std_logic_vector(31 downto 0);
    signal cnt_idle_not_header : std_logic_vector(31 downto 0);

----------------begin event_counter------------------------
begin

    reset_data                          <= not i_reset_data_n;
    reset_dma                           <= not i_reset_dma_n;
    o_event_data                        <= r_ram_data;
    o_all_done(0)                       <= tag_fifo_empty;
    o_all_done(NLINKS downto 1)         <= link_fifo_empty;
    o_fifos_full(NLINKS - 1 downto 0)   <= fifos_full_reg;
    o_fifos_full(NLINKS)                <= i_dmamemhalffull;
    o_fifo_almost_full                  <= link_fifo_almost_full;

    o_cnt_tag_fifo_full <= cnt_tag_fifo_full;
    o_cnt_link_fifo_almost_full <= cnt_link_fifo_almost_full;
    o_cnt_ram_full <= cnt_ram_full;
    o_cnt_stream_fifo_full <= cnt_stream_fifo_full;
    o_cnt_dma_halffull <= cnt_dma_halffull;
    o_cnt_dc_link_fifo_full <= cnt_dc_link_fifo_full;
    o_cnt_skip_event_dma <= cnt_skip_event_dma;
    o_cnt_idle_not_header <= cnt_idle_not_header;

    -- delay writeregs
    process(i_clk_dma, i_reset_dma_n)
    begin
        if( i_reset_dma_n = '0' ) then
            wen_reg     <= '0';
            get_n_words <= (others => '0');
            link_mask_n <= (others => '0');
        elsif(rising_edge(i_clk_dma)) then
            wen_reg     <= i_wen_reg;
            get_n_words <= i_get_n_words;
            link_mask_n <= i_link_mask_n;
        end if;
    end process;

    -- count dma overflow signals
    process(i_clk_dma, i_reset_dma_n)
        -- read add size of ram
        variable diff : std_logic_vector(9 - 1 downto 0);
    begin
        if( i_reset_dma_n = '0' ) then
            cnt_tag_fifo_full <= (others => '0');
            cnt_ram_full <= (others => '0');
            cnt_stream_fifo_full <= (others => '0');
            cnt_dma_halffull <= (others => '0');
        elsif(rising_edge(i_clk_dma)) then
            if ( tag_fifo_full = '1' ) then
                cnt_tag_fifo_full <= cnt_tag_fifo_full + '1';
            end if;

            -- TODO fix me
            --if ( w_ram_add >= r_ram_add ) then
            --    diff := w_ram_add - r_ram_add;
            -- else
            --    diff := r_ram_add - w_ram_add;
            --end if;

            --if ( diff(9 - 1) = '1' ) then
            --    cnt_ram_full <= cnt_ram_full + '1';
            -- end if;

            if ( stream_wfull = '1' ) then
                cnt_stream_fifo_full <= cnt_stream_fifo_full + '1';
            end if;

            if ( i_dmamemhalffull = '1' ) then
                cnt_dma_halffull <= cnt_dma_halffull + '1';
            end if;
        end if;
    end process;

    -- count data overflow signal
    process(i_clk_dma, i_reset_dma_n)
    begin
        if ( i_reset_dma_n = '0' ) then
            cnt_dc_link_fifo_full <= (others => '0');
            cnt_link_fifo_almost_full <= (others => '0');
        elsif(rising_edge(i_clk_dma)) then
    --        link_fifo_full : FOR i in 0 to NLINKS - 1 LOOP
                if ( fifos_full_reg(NLINKS - 1 downto 0) /= all_zero ) then
                    -- for now we only count if one is full
                    cnt_dc_link_fifo_full <= cnt_dc_link_fifo_full + '1';
                end if;
    --      END LOOP link_fifo_full;
    -- TODO: only for all at the moment
                if ( link_fifo_almost_full(NLINKS - 1 downto 0) /= all_zero ) then
                    cnt_link_fifo_almost_full <= cnt_link_fifo_almost_full + '1';
                end if;
        end if;
    end process;

    -- generate fifos per link
    buffer_link_fifos: FOR i in 0 to NLINKS - 1 GENERATE

        process(i_clk_data, i_reset_data_n)
        begin
            if ( i_reset_data_n = '0' ) then
                sync_fifo_data(i) <= (others => '0');
                sync_fifo_i_wrreq(i) <= '0';
            elsif ( rising_edge(i_clk_data) ) then
                sync_fifo_data(i) <= i_link_data(31 + i * 32 downto i * 32) & i_link_datak(3 + i * 4 downto i * 4);
                if ( i_link_data(31 + i * 32 downto i * 32) = x"000000BC" and i_link_datak(3 + i * 4 downto i * 4) = "0001" ) then
                    sync_fifo_i_wrreq(i) <= '0';
                else
                    sync_fifo_i_wrreq(i) <= '1';
                end if;
            end if;
        end process;

        -- delay signals from e_fifo (timing)
        process(i_clk_dma, i_reset_dma_n)
        begin
            if( i_reset_dma_n = '0' ) then
                fifos_full_reg(i)       <= '0';
                link_fifo_usedw_reg(i)  <= '0';
            elsif(rising_edge(i_clk_dma)) then
                fifos_full_reg(i)       <= fifos_full(i);
                link_fifo_usedw_reg(i)  <= link_fifo_usedw(i);
            end if;
        end process;

        e_sync_fifo : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => 6,
            DATA_WIDTH  => 36--,
        )
        port map (
            data        => sync_fifo_data(i),
            wrreq       => sync_fifo_i_wrreq(i),
            rdreq       => not sync_fifo_empty(i),
            wrclk       => i_clk_data,
            rdclk       => i_clk_dma,
            q           => sync_fifo_q(i),
            rdempty     => sync_fifo_empty(i),
            aclr        => '0'--,
        );

        e_link_to_fifo : entity work.link_to_fifo
        generic map(
            W => 32--,
        )
        port map(
            i_link_data         => sync_fifo_q(i)(35 downto 4),
            i_link_datak        => sync_fifo_q(i)(3 downto 0),
            i_fifo_almost_full  => link_fifo_almost_full(i),
            i_sync_fifo_empty   => sync_fifo_empty(i),
            o_fifo_data         => link_data_f(i)(35 downto 0),
            o_fifo_wr           => link_fifo_wren(i),
            o_cnt_skip_data     => o_cnt_skip_link_data,
            i_reset_n           => i_reset_dma_n,
            i_clk               => i_clk_dma--,
        );

        -- sop
        link_data_f(i)(36) <= '1' when ( link_data_f(i)(3 downto 0) = "0001" and link_data_f(i)(11 downto 4) = x"BC" ) else '0';
        -- eop
        link_data_f(i)(37) <= '1' when ( link_data_f(i)(3 downto 0) = "0001" and link_data_f(i)(11 downto 4) = x"9C" ) else '0';

        e_fifo : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => LINK_FIFO_ADDR_WIDTH,
            DATA_WIDTH  => 38--,
        )
        port map (
            data        => link_data_f(i),
            wrreq       => link_fifo_wren(i),
            rdreq       => link_fifo_ren(i),--_reg(i),
            wrclk       => i_clk_dma,
            rdclk       => i_clk_dma,
            q           => link_dataq_f(i),
            rdempty     => link_fifo_empty(i),
            rdusedw     => open,
            wrfull      => fifos_full(i),
            wrusedw     => link_fifo_usedw(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1 downto i * LINK_FIFO_ADDR_WIDTH),
            aclr        => reset_dma--,
        );

        sop(i) <= link_dataq_f(i)(36);
        shop(i) <= '1' when link_dataq_f(i)(37 downto 36) = "00" and link_dataq_f(I)(31 downto 26) = "111111" else '0';
        eop(i) <= link_dataq_f(i)(37);

        process(i_clk_dma, i_reset_dma_n)
        begin
            if(i_reset_dma_n = '0') then
                link_fifo_almost_full(i)       <= '0';
            elsif(rising_edge(i_clk_dma)) then
                if(link_fifo_usedw_reg(i * LINK_FIFO_ADDR_WIDTH + LINK_FIFO_ADDR_WIDTH - 1) = '1') then
                    link_fifo_almost_full(i)   <= '1';
                else
                    link_fifo_almost_full(i)   <= '0';
                end if;
            end if;
        end process;

    END GENERATE buffer_link_fifos;

    e_ram_32_256 : entity work.ip_ram
    generic map (
        ADDR_WIDTH_A    => 12,
        ADDR_WIDTH_B    => 9,
        DATA_WIDTH_A    => 32,
        DATA_WIDTH_B    => 256--,
    )
    port map (
        address_a       => w_ram_add,
        address_b       => r_ram_add,
        clock_a         => i_clk_dma,
        clock_b         => i_clk_dma,
        data_a          => w_ram_data,
        data_b          => (others => '0'),
        wren_a          => w_ram_en,
        wren_b          => '0',
        q_a             => open,
        q_b             => r_ram_data--,
    );

    e_tagging_fifo_event : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH      => 12,
        DATA_WIDTH      => 12--,
    )
    port map (
        data            => w_fifo_data,
        wrreq           => w_fifo_en,
        rdreq           => r_fifo_en,
        clock           => i_clk_dma,
        q               => r_fifo_data,
        full            => tag_fifo_full,
        empty           => tag_fifo_empty,
        sclr            => reset_dma--,
    );

    stream_in_rempty <= link_fifo_empty or not link_mask_n;



    generate_stream_merger : IF USE_ALIGNMENT = 0 GENERATE
        e_stream : entity work.swb_stream_merger
        generic map (
            N => NLINKS--,
        )
        port map (
            i_rdata                 => link_dataq_f,
            i_rsop                  => sop,
            i_reop                  => eop,
            i_rempty                => stream_in_rempty,
            o_rack                  => link_fifo_ren,

            o_wdata(35 downto 0)    => stream_wdata,
            i_wfull                 => stream_wfull,
            o_we                    => stream_we,

            i_reset_n               => i_reset_dma_n,
            i_clk                   => i_clk_dma--,
        );

        e_stream_fifo : entity work.ip_scfifo
        generic map (
            ADDR_WIDTH => 8,
            DATA_WIDTH => 36--,
        )
        port map (
            q               => stream_rdata,
            empty           => stream_rempty,
            rdreq           => stream_rack,
            data            => stream_wdata,
            full            => stream_wfull,
            wrreq           => stream_we,
            sclr            => reset_dma,
            clock           => i_clk_dma--,
        );

        link_data <= stream_rdata(35 downto 4);
        link_datak <= stream_rdata(3 downto 0);

        link_header <=
            '1' when link_datak = "0001" and link_data(7 downto 0) = x"BC"
            else '0';
        link_trailer <=
            '1' when link_datak = "0001" and link_data(7 downto 0) = x"9C"
            else '0';
    end generate;



    time_alignment : if USE_ALIGNMENT = 1 GENERATE

        ---- reg for link FIFO outputs (timing)
        --reg_link_fifos: FOR i in 0 to NLINKS - 1 GENERATE
        --    process(i_clk_dma, i_reset_dma_n)
        --    begin
        --    if ( i_reset_dma_n /= '1' ) then
        --        link_dataq_f_reg(i)    <= (others => '0');
        --        link_fifo_empty_reg(i) <= '0';
        --        sop_reg(i)             <= '0';
        --        eop_reg(i)             <= '0';
        --        shop_reg(i)            <= '0';
        --        link_fifo_ren_reg(i)   <= '0';
        --        --
        --    elsif rising_edge(i_clk_dma) then
        --        link_dataq_f_reg(i)    <= link_dataq_f(i);
        --        link_fifo_ren_reg(i)   <= link_fifo_ren(i);
        --        link_fifo_empty_reg(i) <= link_fifo_empty(i);
        --        sop_reg(i)             <= sop(i);
        --        eop_reg(i)             <= eop(i);
        --        shop_reg(i)            <= shop(i);
        --    end if;
        --    end process;
        --END GENERATE reg_link_fifos;

        e_time_merger : entity work.time_merger
            generic map (
            W => 64+12,
            TREE_DEPTH_w => TREE_w,
            TREE_DEPTH_r => TREE_r,
            N => NLINKS--,
        )
        port map (
            -- input streams
            i_rdata                 => link_dataq_f,--link_dataq_f_reg,
            i_rsop                  => sop,--sop_reg,
            i_reop                  => eop,--eop_reg,
            i_rshop                 => shop,--shop_reg,
            i_rempty                => link_fifo_empty,--link_fifo_empty_reg,
            i_link                  => 1, -- which link should be taken to check ts etc.
            i_mask_n                => link_mask_n,
            o_rack                  => link_fifo_ren,--link_fifo_ren,

            -- output stream
            o_rdata(37 downto 0)   => time_merger_hit,
            i_ren                   => not time_rempty and not stream_wfull,
            o_empty                 => time_rempty,

            -- error outputs

            i_reset_n               => i_reset_dma_n,
            i_clk                   => i_clk_dma--,
        );

        -- link number
        link_number                <= time_merger_hit(37 downto 32);
        -- hit
        stream_wdata(31 downto 0)  <= time_merger_hit(31 downto 0);
        -- header info
        stream_wdata(33 downto 32) <=   "01" when time_merger_hit(37 downto 32) = pre_marker else
                                        "11" when time_merger_hit(37 downto 32) = err_marker else
                                        "10" when time_merger_hit(37 downto 32) = tr_marker else
                                        "00";
        stream_wdata(35 downto 34)  <= "00";
        stream_rdata(35 downto 34)  <= "00";

        e_stream_fifo : entity work.ip_scfifo
        generic map (
            ADDR_WIDTH => 8,
            DATA_WIDTH => 34--,
        )
        port map (
            q               => stream_rdata(33 downto 0),
            empty           => stream_rempty,
            rdreq           => stream_rack,
            data            => stream_wdata(33 downto 0),
            full            => stream_wfull,
            wrreq           => not time_rempty and not stream_wfull,
            sclr            => reset_dma,
            clock           => i_clk_dma--,
        );

        link_data <= stream_rdata(31 downto 0);
        link_header <= '1' when stream_rdata(33 downto 32) = "01" else '0';
        link_trailer <= '1' when stream_rdata(33 downto 32) = "10" else '0';
        -- TODO: handle errors, at the moment they are sent out at the end of normal events
        link_error <= '1' when stream_rdata(33 downto 32) = "11" and stream_rdata(7 downto 0) = x"DC" else '0';

    END GENERATE time_alignment;

    stream_rack <=
        '1' when ( event_tagging_state = bank_data and stream_rempty = '0' ) else
        '1' when ( event_tagging_state = EVENT_IDLE and stream_rempty = '0' and link_header = '0' ) else
        '0';

    -- write link data to event ram
    process(i_clk_dma, i_reset_dma_n)
    begin
    if ( i_reset_dma_n = '0' ) then
        data_flag           <= '0';
        cur_size_add        <= (others => '0');
        cur_bank_size_add   <= (others => '0');
        cur_bank_length_add <= (others => '0');
        w_ram_add_reg       <= (others => '0');
        last_event_add      <= (others => '0');
        align_event_size    <= (others => '0');

        -- ram and tagging fifo write signals
        w_ram_en            <= '0';
        w_ram_data          <= (others => '0');
        w_ram_add           <= (others => '1');
        w_fifo_en           <= '0';
        w_fifo_data         <= (others => '0');

        -- midas signals
        event_id            <= x"0001";
        trigger_mask        <= (others => '0');
        serial_number       <= x"00000001";
        time_tmp            <= (others => '0');
        flags               <= x"00000001";
        type_bank           <= x"00000006"; -- MIDAS Bank Type TID_DWORD

        -- for size counting in bytes
        bank_size_cnt       <= (others => '0');
        event_size_cnt      <= (others => '0');
        cnt_idle_not_header <= (others => '0');

        -- state machine singals
        event_tagging_state <= EVENT_IDLE;

        --
    elsif rising_edge(i_clk_dma) then
        flags           <= x"00000011";
        trigger_mask    <= (others => '0');
        event_id        <= x"0001";
        type_bank       <= x"00000006";
        w_ram_en        <= '0';
        w_fifo_en       <= '0';

        if ( event_tagging_state /= EVENT_IDLE ) then
            -- count time for midas event header
            time_tmp <= time_tmp + '1';
        end if;

        case event_tagging_state is
        when EVENT_IDLE =>
            -- start if at least one not masked link has data
            if ( stream_rempty = '0' and link_header = '1' ) then
                event_tagging_state <= event_head;
            elsif ( stream_rempty = '0' and link_header = '0' ) then
                cnt_idle_not_header <= cnt_idle_not_header + 1;
            end if;

        when event_head =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= trigger_mask & event_id;
            last_event_add      <= w_ram_add + 1;
            event_tagging_state <= event_num;

        when event_num =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= serial_number;
            event_tagging_state <= event_tmp;

        when event_tmp =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= time_tmp;
            event_tagging_state <= event_size;

        when event_size =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_tagging_state <= bank_size;
            cur_size_add        <= w_ram_add + 1;

        when bank_size =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_size_cnt      <= event_size_cnt + 4;
            cur_bank_size_add   <= w_ram_add + 1;
            event_tagging_state <= bank_flags;

        when bank_flags =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_add_reg       <= w_ram_add + 1;
            w_ram_data          <= flags;
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= bank_name;

        when bank_name =>

            -- here we check if the link is masked and if the current fifo is empty
            -- check for mupix or mutrig data header
            if ( stream_rempty = '0' and link_header = '1' ) then
                data_flag <= '1';
                w_ram_en    <= '1';
                w_ram_add   <= w_ram_add_reg + 1;
                -- MIDAS expects bank names in ascii:
                --w_ram_data <=   work.util.to_slv(work.util.to_hstring(link_data(11 downto 8))) &
                --                work.util.to_slv(work.util.to_hstring(link_data(15 downto 12))) &
                --                work.util.to_slv(work.util.to_hstring(link_data(19 downto 16))) &
                --                work.util.to_slv(work.util.to_hstring(link_data(23 downto 20)));
                if(link_data(23 downto 8) = x"FEB0") then
                    w_ram_data <= x"30424546";
                elsif(link_data(23 downto 8) = x"FEB1") then
                    w_ram_data <= x"31424546";
                elsif(link_data(23 downto 8) = x"FEB2") then
                    w_ram_data <= x"32424546";
                elsif(link_data(23 downto 8) = x"FEB3") then
                    w_ram_data <= x"33424546";
                else
                    w_ram_data <= x"34424546";
                end if;
                event_size_cnt      <= event_size_cnt + 4;

                event_tagging_state <= bank_type;
            end if;

        when bank_type =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= type_bank;
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= bank_length;

        when bank_length =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            event_size_cnt      <= event_size_cnt + 4;
            cur_bank_length_add <= w_ram_add + 1;
            event_tagging_state <= bank_data;

        when bank_data =>
            -- check again if the fifo is empty
            if ( stream_rempty = '0' ) then
                w_ram_en            <= '1';
                w_ram_add           <= w_ram_add + 1;
                w_ram_data          <= link_data;
                event_size_cnt      <= event_size_cnt + 4;
                bank_size_cnt       <= bank_size_cnt + 4;
                if ( link_trailer = '1' ) then
                    -- check if the size of the bank data is in 64 bit if not add a word
                    -- this word is not counted to the bank size
                    if ( bank_size_cnt(2 downto 0) = "000" ) then
                        event_tagging_state <= set_algin_word;
                    else
                        event_tagging_state <= bank_set_length;
                        w_ram_add_reg       <= w_ram_add + 1;
                    end if;
                end if;
            end if;

        when set_algin_word =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_add_reg       <= w_ram_add + 1;
            w_ram_data          <= x"AFFEAFFE";
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= bank_set_length;

        when bank_set_length =>
            w_ram_en            <= '1';
            w_ram_add           <= cur_bank_length_add;
            w_ram_data          <= bank_size_cnt;
            bank_size_cnt       <= (others => '0');
--            if ( stream_rempty = '1' ) then
            event_tagging_state <= trailer_name;
--            else
--                event_tagging_state <= bank_name;
--            end if;

        when trailer_name =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add_reg + 1;
            w_ram_data          <= x"454b4146"; -- FAKE in ascii
            data_flag           <= '0';
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= trailer_type;

        when trailer_type =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= type_bank;
            event_size_cnt      <= event_size_cnt + 4;
            event_tagging_state <= trailer_length;

        when trailer_length =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_add_reg       <= w_ram_add + 1;
            w_ram_data          <= (others => '0');
            -- reg trailer length add
            event_size_cnt      <= event_size_cnt + 4;
            -- write at least one AFFEAFFE
            align_event_size    <= w_ram_add + 1 - last_event_add;
            event_tagging_state <= trailer_data;

        when trailer_data =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add + 1;
            w_ram_data          <= x"AFFEAFFE";
            align_event_size    <= align_event_size + 1;
            -- align to DMA word (32 bytes) boundary
            if ( align_event_size(2 downto 0) + '1' = "000" ) then
                event_tagging_state <= trailer_set_length;
            else
                bank_size_cnt   <= bank_size_cnt + 4;
                event_size_cnt  <= event_size_cnt + 4;
            end if;

        when trailer_set_length =>
            w_ram_en            <= '1';
            w_ram_add           <= w_ram_add_reg;
            w_ram_add_reg       <= w_ram_add;
            -- bank length: size in bytes of the following data
            w_ram_data          <= bank_size_cnt;
            bank_size_cnt       <= (others => '0');
            event_tagging_state <= event_set_size;

        when event_set_size =>
            w_ram_en            <= '1';
            w_ram_add           <= cur_size_add;
            -- Event Data Size: The event data size contains the size of the event in bytes excluding the event header
            w_ram_data          <= event_size_cnt;
            event_tagging_state <= bank_set_size;

        when bank_set_size =>
            w_ram_en            <= '1';
            w_ram_add           <= cur_bank_size_add;
            -- All Bank Size : Size in bytes of the following data banks including their bank names
            w_ram_data          <= event_size_cnt - 8;
            event_size_cnt      <= (others => '0');
            event_tagging_state <= write_tagging_fifo;

        when write_tagging_fifo =>
            w_fifo_en           <= '1';
            w_fifo_data         <= w_ram_add_reg;
            last_event_add      <= w_ram_add_reg;
            w_ram_add           <= w_ram_add_reg - 1;
            event_tagging_state <= EVENT_IDLE;
            cur_bank_length_add <= (others => '0');
            serial_number       <= serial_number + '1';

        when others =>
            event_tagging_state <= EVENT_IDLE;

        end case;

    end if;
    end process;


    -- dma end of events, count events and write control
    process(i_clk_dma, i_reset_dma_n)
    begin
    if ( i_reset_dma_n = '0' ) then
        o_event_wren        <= '0';
        o_endofevent        <= '0';
        o_state_out         <= x"0";
        cnt_skip_event_dma  <= (others => '0');
        o_done              <= '0';
        r_fifo_en           <= '0';
        r_ram_add           <= (others => '1');
        event_last_ram_add  <= (others => '0');
        event_counter_state <= waiting;
        word_counter        <= (others => '0');
        --
    elsif rising_edge(i_clk_dma) then

        o_done          <= '0';
        r_fifo_en       <= '0';
        o_event_wren    <= '0';
        o_endofevent    <= '0';

        if ( wen_reg = '0' ) then
            word_counter <= (others => '0');
        end if;

        if ( wen_reg = '1' and word_counter >= i_get_n_words ) then
            o_done <= '1';
        end if;

        case event_counter_state is
        when waiting =>
                o_state_out             <= x"A";
                if (tag_fifo_empty = '0') then
                    r_fifo_en           <= '1';
                    event_last_ram_add  <= r_fifo_data(11 downto 3);
                    r_ram_add           <= r_ram_add + '1';
                    event_counter_state <= get_data;
                end if;

        when get_data =>
                o_state_out             <= x"B";
                if ( i_dmamemhalffull = '1' or ( i_get_n_words /= (i_get_n_words'range => '0') and word_counter >= i_get_n_words ) ) then
                    event_counter_state <= skip_event;
                    cnt_skip_event_dma  <= cnt_skip_event_dma + '1';
                else
                    o_event_wren        <= wen_reg;
                    o_endofevent        <= '1'; -- begin of event
                    word_counter        <= word_counter + '1';
                    event_counter_state <= runing;
                end if;
                r_ram_add       <= r_ram_add + '1';

        when runing =>
                o_state_out     <= x"C";
                o_event_wren    <= wen_reg;
                word_counter    <= word_counter + '1';
                if(r_ram_add = event_last_ram_add - '1') then
                    event_counter_state	<= waiting;
                else
                    r_ram_add <= r_ram_add + '1';
                end if;

        when skip_event =>
                o_state_out <= x"E";
                if(r_ram_add = event_last_ram_add - '1') then
                    event_counter_state	<= waiting;
                else
                    r_ram_add <= r_ram_add + '1';
                end if;

        when others =>
                o_state_out <= x"D";
                event_counter_state	<= waiting;

        end case;

    end if;
    end process;

end architecture;
