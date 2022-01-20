library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


-- merge packets delimited by SOP and EOP from N input streams
entity time_merger_v3 is
generic (
    W               : positive := 64+12;
    TIMEOUT         : std_logic_vector(31 downto 0) := x"FFFFFFFF";
    TREE_DEPTH_w    : positive := 8;
    TREE_DEPTH_r    : positive := 8;
    g_NLINKS_DATA   : positive := 12;
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE       : std_logic_vector(7 downto 0) := x"01"--;
);
port (
    -- input streams
    i_rdata         : in    work.util.slv34_array_t(g_NLINKS_DATA - 1 downto 0);
    i_rsop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- start of packet (SOP)
    i_reop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- end of packet (EOP)
    i_rshop         : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- sub header of packet (SHOP)
    i_rempty        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_mask_n        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_rack          : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0); -- read ACK

    -- output stream
    o_rdata         : out   std_logic_vector(W-1 downto 0);
    o_rdata_debug   : out   std_logic_vector(37 downto 0);
    i_ren           : in    std_logic;
    o_empty         : out   std_logic;
    o_rdempty_debug : out   std_logic;

    -- error outputs
    o_error_pre     : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_error_sh      : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_error_gtime   : out   std_logic_vector(1 downto 0);
    o_error_shtime  : out   std_logic;

    -- counters
    -- cnt_gtime1_error;
    -- cnt_gtime2_error;
    -- cnt_shtime_error;
    -- wait_cnt_pre;
    -- wait_cnt_sh;
    -- wait_cnt_merger;
    o_counters      : out   work.util.slv32_array_t(5 downto 0);

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of time_merger_v3 is

    -- constants
    constant check_zeros : std_logic_vector(g_NLINKS_DATA - 1 downto 0) := (others => '0');
    constant check_ones  : std_logic_vector(g_NLINKS_DATA - 1 downto 0) := (others => '1');

    -- state machine
    type merge_state_type is (wait_for_pre, compare_time1, compare_time2, wait_for_sh, error_state, merge_hits, get_time1, get_time2, trailer, wait_for_sh_written);
    signal merge_state : merge_state_type;
    type sheader_time_array_t is array (g_NLINKS_DATA - 1 downto 0) of std_logic_vector(9 downto 0);
    signal sheader_time : sheader_time_array_t;
    signal merger_state_signal : std_logic;
    signal shtime : std_logic_vector(9 downto 0);
    signal wait_cnt_pre, wait_cnt_sh, wait_cnt_merger : std_logic_vector(31 downto 0);
    signal header_trailer : std_logic_vector(37 downto 0);
    signal sop_wait, shop_wait, shop_or_trailer_wait, eop_wait, time_wait : std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    signal gtime1, gtime2 : work.util.slv34_array_t(g_NLINKS_DATA - 1 downto 0);
    signal wait_cnt : std_logic_vector(3 downto 0);

    -- error signals
    signal error_gtime1, error_gtime2, error_shtime, error_merger : std_logic;
    signal header_trailer_we : std_logic_vector(1 downto 0);
    signal error_pre, error_sh : std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    signal check_link : integer range 0 to g_NLINKS_DATA - 1;

    -- merger tree
    type fifo_width_t is array (6 downto 0) of integer;
    constant read_width : fifo_width_t := (W, 32+6, 32+6, 32+6, 32+6, 32+6, 32+6);
    constant write_width : fifo_width_t := (32+6, 32+6, 32+6, 32+6, 32+6, 32+6, 32+6);
    constant generate_fifos : fifo_width_t := (1, 2, 4, 8, 16, 32, 64);

    -- layer0
    signal data_0, q_0 : work.util.slv38_array_t(generate_fifos(0) - 1 downto 0);
    signal rdreq_0, wrreq_0, rdempty_0, wrfull_0, reset_0 : std_logic_vector(generate_fifos(0) - 1 downto 0);
    signal mask_n_0 : std_logic_vector(generate_fifos(0) - 1 downto 0) := (others => '0');
    -- layer1
    signal q_1 : work.util.slv38_array_t(generate_fifos(1) - 1 downto 0);
    signal rdempty_1, rdreq_1, mask_n_1 : std_logic_vector(generate_fifos(1) - 1 downto 0);
    -- layer2
    signal q_2 : work.util.slv38_array_t(generate_fifos(2) - 1 downto 0);
    signal rdempty_2, rdreq_2, mask_n_2 : std_logic_vector(generate_fifos(2) - 1 downto 0);
    -- layer3
    signal q_3 : work.util.slv38_array_t(generate_fifos(3) - 1 downto 0);
    signal rdempty_3, rdreq_3, mask_n_3 : std_logic_vector(generate_fifos(3) - 1 downto 0);
    -- layer4
    signal q_4 : work.util.slv38_array_t(generate_fifos(4) - 1 downto 0);
    signal rdempty_4, rdreq_4, mask_n_4 : std_logic_vector(generate_fifos(4) - 1 downto 0);
    -- layer5
    signal q_5 : work.util.slv38_array_t(generate_fifos(5) - 1 downto 0);
    signal rdempty_5, rdreq_5, mask_n_5 : std_logic_vector(generate_fifos(5) - 1 downto 0);
    -- layer6
    signal full_6  : std_logic_vector(generate_fifos(6) - 1 downto 0);
    signal alignment_done : std_logic := '0';
    signal last_layer_state : std_logic_vector(7 downto 0);
    signal merger_finish : std_logic_vector(g_NLINKS_DATA - 1 downto 0) := (others => '0');

    -- debug signals
    signal rdata_last_layer : std_logic_vector(W - 1 downto 0);
    signal rdata_last_layer_debug : std_logic_vector(37 downto 0);
    signal rdata_hit_time : std_logic_vector(4 * 8 - 1 downto 0);

    -- counters
    signal cnt_gtime1_error, cnt_gtime2_error, cnt_shtime_error : std_logic_vector(31 downto 0);

begin

    -- ports out
    o_error_gtime(0)    <= error_gtime1;
    o_error_gtime(1)    <= error_gtime2;
    o_error_shtime      <= error_shtime;
    o_error_pre         <= error_pre;
    o_error_sh          <= error_sh;
    o_rdata             <= rdata_last_layer;
    o_rdata_debug       <= rdata_last_layer_debug;
    o_counters(0)       <= cnt_gtime1_error;
    o_counters(1)       <= cnt_gtime2_error;
    o_counters(2)       <= cnt_shtime_error;
    o_counters(3)       <= wait_cnt_pre;
    o_counters(4)       <= wait_cnt_sh;
    o_counters(5)       <= wait_cnt_merger;

    -- debug signals
    gen_hit_data : FOR i in 0 to 7 GENERATE
        rdata_hit_time(i*4 + 3 downto i*4) <= rdata_last_layer(i*38 + 31 downto i*38 + 28);
    END GENERATE;

    -- TODO: ask Alex
    gen_header_state : FOR i in 0 to g_NLINKS_DATA - 1 GENERATE
        sop_wait(i) <=  '1' when i_mask_n(i) = '0' else
                        i_rsop(i) when i_rempty(i) = '0' else '0';
        shop_wait(i)<=  '1' when i_mask_n(i) = '0' else
                        i_rshop(i) when i_rempty(i) = '0' else '0';
        eop_wait(i) <=  '1' when i_mask_n(i) = '0' else
                        i_reop(i) when i_rempty(i) = '0' else '0';
        shop_or_trailer_wait(i) <= shop_wait(i) or eop_wait(i);
        time_wait(i)<=  '1' when i_mask_n(I) = '0' else
                        '1' when ( merge_state = get_time1 or merge_state = get_time2 ) and i_rempty(I) = '0' else
                        '0';
        o_rack(i)   <=  '0' when i_mask_n(i) = '0' else
                        '1' when merge_state = wait_for_pre and sop_wait = check_ones and full_6(0) = '0' and i_rempty(i) = '0' else
                        '1' when merge_state = wait_for_pre and i_rsop(i) = '0' and i_rempty(i) = '0' else
                        '1' when merge_state = get_time1 and time_wait = check_ones and i_rempty(i) = '0' else
                        '1' when merge_state = get_time2 and time_wait = check_ones and i_rempty(i) = '0' else
                        -- TODO: here we should wait for the shop_wait but for the int run we dont care 
                        --'1' when merge_state = wait_for_sh and shop_wait = check_ones  and full_6(0) = '0' else
                        '1' when merge_state = wait_for_sh and shop_or_trailer_wait = check_ones and i_reop(i) = '0' and full_6(0) = '0' and i_rempty(i) = '0' else
                        -- TODO: also again for the int run we check if the word is not a trailer
                        '1' when merge_state = wait_for_sh and i_rshop(i) = '0' and i_reop(i) = '0' and i_rempty(i) = '0' else
                        '1' when merge_state = trailer and full_6(0) = '0' and i_rempty(i) = '0' else
                        '1' when merge_state = merge_hits and sop_wait(i) = '0' and shop_wait(i) = '0' and eop_wait(i) = '0' and i_rempty(i) = '0' and wrfull_0(i) = '0' else
                        '0';
        merger_finish(i) <= '0' when sop_wait(i) = '0' and shop_wait(i) = '0' and eop_wait(i) = '0' else '1';
    END GENERATE;

    merger_state_signal <= '1' when merge_state = merge_hits else '0';

    gen_write_0 : FOR i in 0 to generate_fifos(0)-1 GENERATE

        gen_zero_layer : IF i < g_NLINKS_DATA GENERATE

            reg_merge_state : process(i_clk, i_reset_n)
            begin
            if ( i_reset_n = '0' ) then
                reset_0(i) <= '1';
            elsif ( rising_edge(i_clk) ) then
                if ( merger_state_signal = '1' ) then
                    reset_0(i) <= '0';
                else
                    reset_0(i) <= '1';
                end if;
            end if;
            end process;

            data_0(i)  <=   tree_padding                                            when merger_finish(i) = '0' and merge_state = merge_hits and eop_wait(i) = '1' else
                            work.mudaq.link_36_to_std(i) & i_rdata(i)(31 downto 0)   when merger_finish(i) = '0' and merge_state = merge_hits else
                            tree_padding                                            when merger_finish(i) = '1' and merge_state = merge_hits else
                            (others => '0');
            wrreq_0(i) <= '1' when merge_state = merge_hits and i_rempty(i) = '0' and wrfull_0(i) = '0' else '0';

--            e_link_fifo : entity work.ip_scfifo
--            generic map(
--                ADDR_WIDTH      => TREE_DEPTH_w,
--                DATA_WIDTH      => write_width(0),
--                RAM_OUT_REG     => "ON",
--                DEVICE          => "Arria 10"--,
--            )
--            port map (
--                sclr    => reset_0(i),
--                data    => data_0(i),
--                clock   => i_clk,
--                rdreq   => rdreq_0(i),
--                wrreq   => wrreq_0(i),
--                q       => q_0(i),
--                empty   => rdempty_0(i),
--                full    => wrfull_0(i)--,
--            );
            
            e_link_fifo : entity work.ip_dcfifo
            generic map(
                ADDR_WIDTH  => TREE_DEPTH_w,
                DATA_WIDTH  => write_width(0),
                DEVICE      => "Arria 10"--,
            )
            port map (
                data        => data_0(i),
                wrreq       => wrreq_0(i),
                rdreq       => rdreq_0(i),
                wrclk       => i_clk,
                rdclk       => i_clk,
                q           => q_0(i),
                rdempty     => rdempty_0(i),
                wrfull      => wrfull_0(i),
                aclr        => reset_0(i)--,
            );

        END GENERATE;

    END GENERATE;

    mask_n_0(g_NLINKS_DATA - 1 downto 0) <= i_mask_n;
    layer_1 : entity work.time_merger_tree_fifo_32_v2
    generic map (
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r, g_NLINKS_DATA => g_NLINKS_DATA,
        r_width => read_width(0), w_width => write_width(1), last_layer => '0',
        compare_fifos => generate_fifos(0), gen_fifos => generate_fifos(1), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        i_data          => q_0,
        i_rdempty       => rdempty_0,
        i_rdreq         => rdreq_1,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_0,
        i_wen_h_t       => "00",
        i_data_h_t      => (others => '0'),

        o_q             => q_1,
        o_rdempty       => rdempty_1,
        o_rdreq         => rdreq_0,
        o_mask_n        => mask_n_1,
        o_layer_state   => open,
        o_wrfull        => open,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_2 : entity work.time_merger_tree_fifo_32_v2
    generic map (
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r, g_NLINKS_DATA => g_NLINKS_DATA,
        r_width => read_width(1), w_width => write_width(2), last_layer => '0',
        compare_fifos => generate_fifos(1), gen_fifos => generate_fifos(2), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        i_data          => q_1,
        i_rdempty       => rdempty_1,
        i_rdreq         => rdreq_2,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_1,
        i_wen_h_t       => "00",
        i_data_h_t      => (others => '0'),

        o_q             => q_2,
        o_rdempty       => rdempty_2,
        o_rdreq         => rdreq_1,
        o_mask_n        => mask_n_2,
        o_layer_state   => open,
        o_wrfull        => open,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_3 : entity work.time_merger_tree_fifo_32_v2
    generic map (
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r, g_NLINKS_DATA => g_NLINKS_DATA,
        r_width => read_width(2), w_width => write_width(3), last_layer => '0',
        compare_fifos => generate_fifos(2), gen_fifos => generate_fifos(3), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        i_data          => q_2,
        i_rdempty       => rdempty_2,
        i_rdreq         => rdreq_3,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_2,
        i_wen_h_t       => "00",
        i_data_h_t      => (others => '0'),

        o_q             => q_3,
        o_rdempty       => rdempty_3,
        o_rdreq         => rdreq_2,
        o_mask_n        => mask_n_3,
        o_layer_state   => open,
        o_wrfull        => open,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_4 : entity work.time_merger_tree_fifo_32_v2
    generic map (
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r, g_NLINKS_DATA => g_NLINKS_DATA,
        r_width => read_width(3), w_width => write_width(4), last_layer => '0',
        compare_fifos => generate_fifos(3), gen_fifos => generate_fifos(4), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        i_data          => q_3,
        i_rdempty       => rdempty_3,
        i_rdreq         => rdreq_4,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_3,
        i_wen_h_t       => "00",
        i_data_h_t      => (others => '0'),

        o_q             => q_4,
        o_rdempty       => rdempty_4,
        o_rdreq         => rdreq_3,
        o_mask_n        => mask_n_4,
        o_layer_state   => open,
        o_wrfull        => open,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_5 : entity work.time_merger_tree_fifo_32_v2
    generic map (
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r, g_NLINKS_DATA => g_NLINKS_DATA,
        r_width => read_width(4), w_width => write_width(5), last_layer => '0',
        compare_fifos => generate_fifos(4), gen_fifos => generate_fifos(5), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        i_data          => q_4,
        i_rdempty       => rdempty_4,
        i_rdreq         => rdreq_5,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_4,
        i_wen_h_t       => "00",
        i_data_h_t      => (others => '0'),

        o_q             => q_5,
        o_rdempty       => rdempty_5,
        o_rdreq         => rdreq_4,
        o_mask_n        => mask_n_5,
        o_layer_state   => open,
        o_wrfull        => open,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_6 : entity work.time_merger_tree_fifo_32_v2
    generic map (
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r, g_NLINKS_DATA => g_NLINKS_DATA,
        r_width => read_width(6), w_width => write_width(6), last_layer => '1',
        compare_fifos => generate_fifos(5), gen_fifos => generate_fifos(6), DATA_TYPE => DATA_TYPE--,
    )
    port map (
        i_data          => q_5,
        i_rdempty       => rdempty_5,
        i_rdreq(0)      => i_ren,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_5,
        i_wen_h_t       => header_trailer_we,
        i_data_h_t      => header_trailer,

        o_last          => rdata_last_layer,
        o_last_link_debug   => rdata_last_layer_debug,
        o_rdempty_debug(0)  => o_rdempty_debug,
        o_rdempty(0)    => o_empty,
        o_rdreq         => rdreq_5,
        o_mask_n        => open,
        o_layer_state(0)=> last_layer_state,
        o_wrfull        => full_6,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    alignment_done <= '1' when last_layer_state = x"8" else '0';

    -- write data
    process(i_clk, i_reset_n)
        variable v_overflow : std_logic_vector(15 downto 0);
        variable TF : std_logic_vector(1 downto 0);
    begin
    if ( i_reset_n = '0' ) then
        merge_state <= wait_for_pre;
        error_pre <= (others => '0');
        error_sh <= (others => '0');
        wait_cnt_pre <= (others => '0');
        wait_cnt_sh <= (others => '0');
        wait_cnt_merger <= (others => '0');
        gtime1 <= (others => (others => '0'));
        gtime2 <= (others => (others => '0'));
        shtime <= (others => '0');
        sheader_time <= (others => (others => '0'));
        error_gtime1 <= '0';
        error_gtime2 <= '0';
        error_shtime <= '0';
        error_merger <= '0';
        header_trailer <= (others => '0');
        header_trailer_we <= "00";
        v_overflow := (others => '0');
        wait_cnt <= (others => '0');
        TF := (others => '0');
        check_link <= 0;
        cnt_gtime1_error <= (others => '0');
        cnt_gtime2_error <= (others => '0');
        cnt_shtime_error <= (others => '0');
        --
    elsif rising_edge(i_clk) then

        header_trailer      <= (others => '0');
        header_trailer_we   <= "00";
        v_overflow  := (others => '0');
        TF          := (others => '0');

        FOR i in 0 to g_NLINKS_DATA - 1 LOOP
            if ( i_mask_n(I) = '1' ) then
                check_link <= I;
                exit;
            end if;
        END LOOP;

        case merge_state is
            when wait_for_pre =>
                -- readout until all fifos have preamble
                if ( sop_wait = check_ones and full_6(0) = '0' ) then
                    merge_state <= get_time1;
                    -- reset signals
                    wait_cnt_pre <= (others => '0');
                    -- send merged data preamble
                    -- sop & preamble & zeros & datak
                    header_trailer(37 downto 32) <= pre_marker;
                    header_trailer(31 downto 26) <= "111010";
                    header_trailer(7 downto 0) <= x"BC";
                    header_trailer_we <= "10";
                else
                    wait_cnt_pre <= wait_cnt_pre + '1';
                end if;

                -- if wait for pre gets timeout
                if ( wait_cnt_pre = TIMEOUT ) then
                    error_pre <= sop_wait;
                    merge_state <= error_state;
                end if;

            -- TODO: change this to one cycle
            when get_time1 =>
                -- get MSB from FPGA time
                if ( time_wait = check_ones ) then
                    merge_state <= compare_time1;
                    gtime1 <= i_rdata;
                end if;

            when compare_time1 =>
                -- compare MSB from FPGA time
                FOR I in g_NLINKS_DATA - 1 downto 0 LOOP
                    if ( gtime1(I) /= gtime1(check_link) and i_mask_n(I) = '1' ) then
                        error_gtime1 <= '1';
                    end if;
                END LOOP;

                -- check if fifo is not full and all links have same time
                -- dont check at the moment error_gtime1 = '0' and
                if ( full_6(0) = '0' ) then
                    merge_state <= get_time2;
                    -- reset signals
                    gtime1 <= (others => (others => '0'));
                    -- send gtime1
                    header_trailer(37 downto 32) <= ts1_marker;
                    header_trailer(31 downto 0) <= gtime1(check_link)(31 downto 0);
                    header_trailer_we <= "10";
                end if;
                -- dont go into error_state at the moment
                if ( error_gtime1 = '1' ) then
                --     merge_state <= error_state;
                    cnt_gtime1_error <= cnt_gtime1_error + '1';
                end if;

            -- TODO: change this to one cycle
            when get_time2 =>
                -- get LSB from FPGA time
                --if ( error_gtime1 = '1' ) then
                --    merge_state <= error_state;
                if ( time_wait = check_ones ) then
                    merge_state <= compare_time2;
                    gtime2 <= i_rdata;
                end if;

            when compare_time2 =>
                -- compare LSB from FPGA time
                FOR I in g_NLINKS_DATA - 1 downto 0 LOOP
                    if ( gtime2(I) /= gtime2(check_link) and i_mask_n(I) = '1' ) then
                        error_gtime2 <= '1';
                    end if;
                END LOOP;

                -- check if fifo is not full and all links have same time
                -- error_gtime2 = '0'
                if ( full_6(0) = '0' ) then
                    merge_state <= wait_for_sh;
                    -- reset signals
                    gtime2 <= (others => (others => '0'));
                    -- send gtime2
                    header_trailer(37 downto 32) <= ts2_marker;
                    header_trailer(31 downto 0) <= gtime2(check_link)(31 downto 0);
                    header_trailer_we <= "10";
                end if;
                -- dont check at the moment
                if ( error_gtime2 = '1' ) then
                --   merge_state <= error_state;
                    cnt_gtime2_error <= cnt_gtime2_error + '1';
                end if;

            when wait_for_sh =>
                -- dont check at the moment
                --if ( error_gtime2 = '1' ) then
                --    merge_state <= error_state;
                --end if;

                -- readout until all fifos have sub header
                -- TODO: here we should wait for shop_wait but for the int run we dont care
                if ( shop_or_trailer_wait = check_ones and full_6(0) = '0' ) then
                    merge_state <= wait_for_sh_written;
                    -- reset signals
                    wait_cnt_sh <= (others => '0');
                    wait_cnt_merger <= (others => '0');
                    -- send merged data sub header
                    -- zeros & sub header & zeros & datak
                    header_trailer(37 downto 32) <= sh_marker;
                    -- send sub header time -- check later if equal
                    header_trailer(31 downto 16) <= i_rdata(check_link)(31 downto 16);
                    if ( DATA_TYPE = x"01" ) then
                        shtime(9 downto 7) <= (others => '0');
                        shtime(6 downto 0) <= i_rdata(check_link)(22 downto 16);
                    elsif ( DATA_TYPE = x"02" ) then
                        shtime <= i_rdata(check_link)(25 downto 16);
                    end if;
                    FOR I in g_NLINKS_DATA - 1 downto 0 LOOP
                        if ( i_mask_n(I) = '1' ) then
                            if ( DATA_TYPE = x"01" ) then
                                sheader_time(I)(9 downto 7) <= (others => '0');
                                sheader_time(I)(6 downto 0) <= i_rdata(I)(22 downto 16);
                            elsif ( DATA_TYPE = x"02" ) then
                                sheader_time(I) <= i_rdata(I)(25 downto 16);
                            end if;
                        end if;
                        -- TODO: For int run we take this one out
                        --v_overflow := v_overflow or i_rdata(I)(15 downto 0);
                    END LOOP;
                    header_trailer(15 downto 0) <= v_overflow;
                    header_trailer_we <= "10";
                else
                    wait_cnt_sh <= wait_cnt_sh + '1';
                end if;

                -- if wait for pre gets timeout
                if ( wait_cnt_sh = TIMEOUT ) then
                    error_sh <= shop_wait;
                    merge_state <= error_state;
                end if;

            -- TODO: Change this to one cycle
            -- NOTE: at the moment we wait here a view cycles to write the 
            -- SubHeader to the 256 bit out
            when wait_for_sh_written =>
                header_trailer_we <= "01";
                if ( wait_cnt = "1000"  ) then
                    wait_cnt <= (others => '0');
                    merge_state <= merge_hits;
                else
                    wait_cnt <= wait_cnt + '1';
                end if;

            when merge_hits =>
                if ( error_shtime = '1' ) then
                    cnt_shtime_error <= cnt_shtime_error + '1';
                    --merge_state <= error_state;
                end if;

                -- check if sheader time is equal
                FOR I in g_NLINKS_DATA - 1 downto 0 LOOP
                    if ( i_rempty(I) = '0' and i_mask_n(I) = '1' and sheader_time(I) /= shtime ) then
                        error_shtime <= '1';
                    end if;
                END LOOP;

                -- TODO use generatic timeout for the moment
                wait_cnt_merger <= wait_cnt_merger + '1';
                if ( wait_cnt_merger = TIMEOUT ) then
                    merge_state <= error_state;
                    error_merger <= '1';
                end if;

                -- change state
                -- TODO error if sh is not there
                -- TODO her we should wait for shop_wait but we dont care
                if ( shop_or_trailer_wait = check_ones and alignment_done = '1' ) then
                    merge_state <= wait_for_sh;
                end if;

                -- TODO error if pre is not there
                -- TODO this should not happen -- error
                if ( sop_wait = check_ones and alignment_done = '1' ) then
                    merge_state <= wait_for_pre;
                end if;

                -- TODO error if trailer is not there
                if ( eop_wait = check_ones and alignment_done = '1' ) then
                    merge_state <= trailer;
                end if;

            when trailer =>
                -- send trailer
                if( full_6(0) = '0' ) then
                    merge_state <= wait_for_pre;
                    -- send trailer
                    header_trailer(37 downto 32) <= tr_marker;
                    if ( DATA_TYPE = x"02" ) then
                        FOR I in g_NLINKS_DATA - 1 downto 0 LOOP
                            TF := i_rdata(I)(9 downto 8);
                        END LOOP;
                        header_trailer(9 downto 8) <= TF;
                    end if;
                    header_trailer(7 downto 0) <= x"9C";
                    header_trailer_we <= "10";
                end if;
                -- reset errors
                error_shtime <= '0';
                error_gtime1 <= '0';
                error_gtime2 <= '0';
                error_merger <= '0';
                error_pre <= (others => '0');
                error_sh <= (others => '0');

            when error_state =>
                -- send error message xxxxxxDC
                -- 12: error gtime1
                -- 13: error gtime2
                -- 14: error shtime
                -- N+14 downto 14: error wait for pre
                header_trailer(37 downto 32) <= err_marker;
                header_trailer(7 downto 0) <= x"DC";
                header_trailer(12) <= error_gtime1;
                header_trailer(13) <= error_gtime2;
                header_trailer(14) <= error_shtime;
                header_trailer(15) <= error_merger;
                if ( error_pre /= check_zeros ) then
                    header_trailer(16) <= '1';
                end if;
                if ( error_sh /= check_zeros ) then
                    header_trailer(17) <= '1';
                end if;
                header_trailer_we <= "10";
                merge_state <= trailer;

            when others =>
                merge_state <= wait_for_pre;

        end case;
        --
    end if;
    end process;

end architecture;
