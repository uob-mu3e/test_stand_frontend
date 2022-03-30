library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


entity time_merger_tree_fifo_32_v2 is
generic (
    TREE_w          : positive  := 8;
    TREE_r          : positive  := 8;
    r_width         : positive  := 64;
    w_width         : positive  := 64;
    compare_fifos   : positive  := 32;
    last_layer      : std_logic := '0';
    g_NLINKS_DATA   : positive  := 12;
    gen_fifos       : positive  := 16;
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE : std_logic_vector(7 downto 0) := x"01"--;
);
port (
    -- input
    i_data          : in  work.util.slv38_array_t(compare_fifos - 1 downto 0) := (others => (others => '1'));
    i_rdempty       : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_rdreq         : in  std_logic_vector(gen_fifos - 1 downto 0);
    i_merge_state   : in  std_logic;
    i_mask_n        : in  std_logic_vector(compare_fifos - 1 downto 0);
    -- "10" = header / "01" = sh padding
    i_wen_h_t       : in  std_logic_vector(1 downto 0);
    i_data_h_t      : in  std_logic_vector(37 downto 0);

    -- output
    o_q             : out work.util.slv38_array_t(gen_fifos - 1 downto 0);
    o_last          : out std_logic_vector(r_width-1 downto 0);
    o_last_link_debug : out std_logic_vector(37 downto 0);
    o_rdempty       : out std_logic_vector(gen_fifos - 1 downto 0) := (others => '1');
    o_rdempty_debug : out std_logic_vector(gen_fifos - 1 downto 0) := (others => '1');
    o_rdreq         : out std_logic_vector(compare_fifos - 1 downto 0);
    o_mask_n        : out std_logic_vector(gen_fifos - 1 downto 0);
    o_layer_state   : out work.util.slv8_array_t(gen_fifos - 1 downto 0);
    o_wrfull        : out std_logic_vector(gen_fifos - 1 downto 0);

    i_reset_n       : in  std_logic;
    i_clk           : in  std_logic--;
);
end entity;

architecture arch of time_merger_tree_fifo_32_v2 is

    -- merger signals
    constant size : integer := compare_fifos/2;

    -- layer states
    constant second_input_not_mask_n : std_logic_vector(7 downto 0) := x"00";
    constant first_input_not_mask_n : std_logic_vector(7 downto 0) := x"01";
    constant a_no_b_padding_state : std_logic_vector(7 downto 0) := x"02";
    constant b_no_a_padding_state : std_logic_vector(7 downto 0) := x"03";
    constant a_smaller_b : std_logic_vector(7 downto 0) := x"04";
    constant b_smaller_a : std_logic_vector(7 downto 0) := x"05";
    constant end_state : std_logic_vector(7 downto 0) := x"08";
    constant last_layer_state : std_logic_vector(7 downto 0) := x"09";
    constant IDEL : std_logic_vector(7 downto 0) := x"FF";

    signal data, q : work.util.slv38_array_t(gen_fifos - 1 downto 0);
    signal layer_state : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal wrreq, wrfull, reset_fifo, wrfull_and_merge_state, both_inputs_rdempty, rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal wrfull_and_merge_state_and_both_inputs_not_rdempty, wrfull_and_merge_state_and_first_input_not_rdempty, wrfull_and_merge_state_and_second_input_not_rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal first_input_mask_n_second_input_not_mask_n, second_input_mask_n_first_input_not_mask_n, a_padding, b_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a_b_padding, b_no_a_padding, a_no_b_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a, b : work.util.slv4_array_t(gen_fifos - 1 downto 0) := (others => (others => '1'));
    signal a_h, b_h : work.util.slv38_array_t(gen_fifos - 1 downto 0) := (others => (others => '1'));
    signal last : std_logic_vector(r_width-1 downto 0);
    signal last_data : std_logic_vector(8 * 32 - 1 downto 0);
    signal last_link : std_logic_vector(8 *  6 - 1 downto 0);
    signal wrfull_last0, wrfull_last1, wrfull_last2, wrfull_s, rdempty_last0, rdempty_last1, rdempty_last2 : std_logic_vector(gen_fifos - 1 downto 0);

    -- for debugging / simulation
    signal t_q, t_data : work.util.slv4_array_t(gen_fifos - 1 downto 0);
    signal t_q_last : std_logic_vector(31 downto 0);
    signal l1 : work.util.slv6_array_t(gen_fifos - 1 downto 0);

begin

    gen_hits:
    FOR i in 0 to gen_fifos - 1 GENERATE
        mupix_data : IF DATA_TYPE = x"01" GENERATE
            a(i)    <= i_data(i)(31 downto 28) when i_mask_n(i) = '1' else (others => '1');
            b(i)    <= i_data(i+size)(31 downto 28) when i_mask_n(i+size) = '1' else (others => '1');
        END GENERATE;
        scifi_data : IF DATA_TYPE = x"02" GENERATE
            a(i)    <= i_data(i)(9 downto 6) when i_mask_n(i) = '1' else (others => '1');
            b(i)    <= i_data(i+size)(9 downto 6) when i_mask_n(i+size) = '1' else (others => '1');
        END GENERATE;

        a_h(i)      <= i_data(i)(37 downto 0) when i_mask_n(i) = '1' else (others => '1');
        b_h(i)      <= i_data(i+size)(37 downto 0) when i_mask_n(i+size) = '1' else (others => '1');

        -- for debugging / simulation
        t_q(i)      <= q(i)(31 downto 28);
        t_data(i)   <= data(i)(31 downto 28);
        l1(i)       <= data(i)(37 downto 32);
    END GENERATE;

    gen_last_layer : if last_layer = '1' generate
        gen_last:
        FOR i in 0 to 7 GENERATE
            last(38 * i + 37 downto 38 * i) <= last_link(6 * i + 5 downto 6 * i) & last_data(32 * i + 31 downto 32 * i);
        END GENERATE;
        t_q_last(31 downto 28) <= last(297 downto 294);
        t_q_last(27 downto 24) <= last(259 downto 256);
        t_q_last(23 downto 20) <= last(221 downto 218);
        t_q_last(19 downto 16) <= last(183 downto 180);
        t_q_last(15 downto 12) <= last(145 downto 142);
        t_q_last(11 downto  8) <= last(107 downto 104);
        t_q_last( 7 downto  4) <= last(69 downto 66);
        t_q_last( 3 downto  0) <= last(31 downto 28);
        o_wrfull               <= wrfull_last2; --(wrfull_last0 and wrfull_last1) or
        wrfull_s               <= wrfull_last2; --(wrfull_last0 and wrfull_last1) or
        o_rdempty              <= rdempty_last2;--(rdempty_last0 and rdempty_last1) or
        o_rdempty_debug        <= rdempty_last2;
    end generate;

    gen_not_last_layer : if last_layer = '0' generate
        o_wrfull    <= wrfull;
        wrfull_s    <= wrfull;
        o_rdempty   <= rdempty;
    end generate;

    o_layer_state   <= layer_state;
    o_q             <= q;
    o_last          <= last;


    gen_tree:
    FOR i in 0 to gen_fifos - 1 GENERATE

        reg_merge_state : process(i_clk, i_reset_n)
        begin
        if ( i_reset_n = '0' ) then
            reset_fifo(i) <= '1';
        elsif ( rising_edge(i_clk) ) then
            if ( i_merge_state = '1' or last_layer = '1' ) then
                reset_fifo(i) <= '0';
            else
                reset_fifo(i) <= '1';
            end if;
        end if;
        end process;

        o_mask_n(i) <= i_mask_n(i) or i_mask_n(i + size);

        gen_last_layer : IF last_layer = '1' and i < g_NLINKS_DATA GENERATE
--            e_last_fifo_data : entity work.ip_dcfifo_v2
--            generic map(
--                g_WADDR_WIDTH => 11,
--                g_WDATA_WIDTH => 32,
--                g_RADDR_WIDTH => 8,
--                g_RDATA_WIDTH => 32*8--,
--            )
--            port map (
--                i_we        => wrreq(i),
--                i_wdata     => data(i)(31 downto 0),
--                o_wfull     => wrfull_last0(i),
--                i_wclk      => i_clk,
--
--                i_rack      => i_rdreq(i),
--                o_rdata     => last_data,
--                o_rempty    => rdempty_last0(i),
--                i_rclk      => i_clk,
--
--                i_reset_n   => not reset_fifo(i)--,
--            );
--
--            e_last_fifo_link : entity work.ip_dcfifo_v2
--            generic map(
--                g_WADDR_WIDTH => 11,
--                g_WDATA_WIDTH => 6,
--                g_RADDR_WIDTH => 8,
--                g_RDATA_WIDTH => 6*8--,
--            )
--            port map (
--                i_we        => wrreq(i),
--                i_wdata     => data(i)(37 downto 32),
--                o_wfull     => wrfull_last1(i),
--                i_wclk      => i_clk,
--
--                i_rack      => i_rdreq(i),
--                o_rdata     => last_link,
--                o_rempty    => rdempty_last1(i),
--                i_rclk      => i_clk,
--
--                i_reset_n   => not reset_fifo(i)--,
--            );

            e_last_fifo_link_debug : entity work.ip_scfifo_v2
            generic map (
                g_ADDR_WIDTH => 11,
                g_DATA_WIDTH => 38,
                g_WREG_N => 1,
                g_RREG_N => 1--,
            )
            port map (
                i_wdata         => data(i),
                i_we            => wrreq(i),
                o_wfull         => wrfull_last2(i),

                o_rdata         => o_last_link_debug,
                o_rempty        => rdempty_last2(i),
                i_rack          => i_rdreq(i),

                i_clk           => i_clk,
                i_reset_n       => not reset_fifo(i)--,
            );
        END GENERATE;

        gen_layer : IF last_layer = '0' and i < g_NLINKS_DATA GENERATE
            e_link_fifo : entity work.ip_scfifo_v2
            generic map (
                g_ADDR_WIDTH => TREE_w,
                g_DATA_WIDTH => w_width,
                g_WREG_N => 1,
                g_RREG_N => 1--,
            )
            port map (
                i_wdata         => data(i),
                i_we            => wrreq(i),
                o_wfull         => wrfull(i),

                o_rdata         => q(i),
                o_rempty        => rdempty(i),
                i_rack          => i_rdreq(i),

                i_clk           => i_clk,
                i_reset_n       => not reset_fifo(i)--,
            );
        END GENERATE;


        -- Tree setup
        -- x => empty, F => padding
        -- [a]              [a]             [a]
        -- [1]  -> [2,1]    [x]  -> [x,2]   [F]  -> [x,2]
        -- [2]              [2]             [2]
        -- [b]              [b]             [b]

        -- TODO: what to do when the FIFO (wrfull(i) = '1') gets full? There need to be a waiting state which does nothing
        wrfull_and_merge_state(i)                               <= '1' when i_merge_state = '1' and wrfull_s(i) = '0' else '0';
        both_inputs_rdempty(i)                                  <= '0' when i_rdempty(i) = '0' and i_rdempty(i+size) = '0' else '1';
        wrfull_and_merge_state_and_both_inputs_not_rdempty(i)   <= '1' when wrfull_and_merge_state(i) = '1' and both_inputs_rdempty(i) = '0' else '0';
        wrfull_and_merge_state_and_first_input_not_rdempty(i)   <= '1' when wrfull_and_merge_state(i) = '1' and i_rdempty(i) = '0' else '0';
        wrfull_and_merge_state_and_second_input_not_rdempty(i)  <= '1' when wrfull_and_merge_state(i) = '1' and i_rdempty(i+size) = '0' else '0';
        first_input_mask_n_second_input_not_mask_n(i)           <= '1' when i_mask_n(i) = '1' and i_mask_n(i+size) = '0' else '0';
        second_input_mask_n_first_input_not_mask_n(i)           <= '1' when i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else '0';
        a_padding(i)                                            <= '1' when a_h(i) = tree_padding else '0';
        b_padding(i)                                            <= '1' when b_h(i) = tree_padding else '0';
        a_b_padding(i)                                          <= '1' when a_padding(i) = '1' and b_padding(i) = '1' else '0';
        a_no_b_padding(i)                                       <= '1' when a_padding(i) = '0' and b_padding(i) = '1' else '0';
        b_no_a_padding(i)                                       <= '1' when a_padding(i) = '1' and b_padding(i) = '0' else '0';

        -- TODO: include sub-header, check backpres., counters etc.
        layer_state(i) <= last_layer_state when i_merge_state = '0' and last_layer = '1' else

                          end_state when a_b_padding(i) = '1' else

                          second_input_not_mask_n when wrfull_and_merge_state_and_first_input_not_rdempty(i) = '1' and first_input_mask_n_second_input_not_mask_n(i) = '1' else
                          first_input_not_mask_n when wrfull_and_merge_state_and_second_input_not_rdempty(i) = '1' and second_input_mask_n_first_input_not_mask_n(i) = '1' else

                          a_no_b_padding_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' else
                          b_no_a_padding_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b_no_a_padding(i) = '1' else

                          a_smaller_b when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= b(i) else
                          b_smaller_a when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i)  > b(i) else

                          IDEL;

        wrreq(i)        <=  '1' when layer_state(i) = last_layer_state and (i_wen_h_t = "01" or i_wen_h_t = "10" ) else
                            '1' when layer_state(i) = end_state and wrfull_s(i) = '0' else
                            '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = first_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = b_no_a_padding_state or layer_state(i) = a_smaller_b or layer_state(i) = b_smaller_a else
                            '0';

        o_rdreq(i)      <=  '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = a_smaller_b else
                            '0';

        o_rdreq(i+size) <=  '1' when layer_state(i) = first_input_not_mask_n or layer_state(i) = b_no_a_padding_state or layer_state(i) = b_smaller_a else
                            '0';

        data(i)         <= i_data_h_t when layer_state(i) = last_layer_state and i_wen_h_t = "10" else
                           tree_padding when layer_state(i) = last_layer_state and i_wen_h_t = "01" else
                           tree_padding when layer_state(i) = end_state else
                           a_h(i) when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = a_smaller_b else
                           b_h(i) when layer_state(i) = first_input_not_mask_n or layer_state(i) = b_no_a_padding_state or layer_state(i) = b_smaller_a else
                           (others => '0');

    END GENERATE;

end architecture;
