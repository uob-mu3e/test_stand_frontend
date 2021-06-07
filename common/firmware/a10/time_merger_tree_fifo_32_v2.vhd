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
    gen_fifos       : positive  := 16--;
);
port (
    -- input
    i_data          : in  work.util.slv38_array_t(compare_fifos - 1 downto 0);
    i_rdempty       : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_rdreq         : in  std_logic_vector(gen_fifos - 1 downto 0);
    i_merge_state   : in  std_logic;
    i_mask_n        : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_wen_h_t       : in  std_logic;
    i_data_h_t      : in  std_logic_vector(37 downto 0);

    -- output
    o_q             : out work.util.slv38_array_t(gen_fifos - 1 downto 0);
    o_last          : out std_logic_vector(r_width-1 downto 0);
    o_rdempty       : out std_logic_vector(gen_fifos - 1 downto 0) := (others => '1');
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

    signal data, data_reg, q, q_reg, q_reg_reg : work.util.slv38_array_t(gen_fifos - 1 downto 0);
    signal layer_state, layer_state_reg : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal wrreq, f_wrreq, f_wrreq_reg, wrfull, reset_fifo, wrfull_and_merge_state, both_inputs_rdempty, rdempty, rdempty_reg, wrfull_reg, rdreq : std_logic_vector(gen_fifos - 1 downto 0);
    signal rdempty_reg_reg, wrfull_reg_reg, rdreq_reg : std_logic_vector(gen_fifos - 1 downto 0);
    signal wrfull_and_merge_state_and_both_inputs_not_rdempty, wrfull_and_merge_state_and_first_input_not_rdempty, wrfull_and_merge_state_and_second_input_not_rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal first_input_mask_n_second_input_not_mask_n, second_input_mask_n_first_input_not_mask_n, a_padding, b_padding, c_padding, d_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a_b_padding, b_no_a_padding, a_no_b_padding, c_d_padding, c_d_no_padding, c_no_d_padding, a_c_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a, b : work.util.slv4_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_h, b_h : work.util.slv38_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal last, last_reg, last_reg_reg : std_logic_vector(r_width-1 downto 0);

    -- for debugging / simulation
    signal t_q, t_data : work.util.slv4_array_t(gen_fifos - 1 downto 0);
    signal t_q_last : std_logic_vector(31 downto 0);
    signal l1 : work.util.slv6_array_t(gen_fifos - 1 downto 0);

begin

    gen_hits:
    FOR i in 0 to gen_fifos - 1 GENERATE
        a(i) <= i_data(i)(31 downto 28);
        b(i) <= i_data(i + size)(31 downto 28);

        a_h(i) <= i_data(i)(37 downto 0);
        b_h(i) <= i_data(i + size)(37 downto 0);

        -- for debugging / simulation
        t_q(i) <= q_reg_reg(i)(31 downto 28);
        t_data(i) <= data(i)(31 downto 28);
        l1(i) <= data(i)(37 downto 32);
    END GENERATE;

    gen_last_layer : if last_layer = '1' generate
        t_q_last(31 downto 28) <= last_reg_reg(297 downto 294);
        t_q_last(27 downto 24) <= last_reg_reg(259 downto 256);
        t_q_last(23 downto 20) <= last_reg_reg(221 downto 218);
        t_q_last(19 downto 16) <= last_reg_reg(183 downto 180);
        t_q_last(15 downto 12) <= last_reg_reg(145 downto 142);
        t_q_last(11 downto  8) <= last_reg_reg(107 downto 104);
        t_q_last( 7 downto  4) <= last_reg_reg(69 downto 66);
        t_q_last( 3 downto  0) <= last_reg_reg(31 downto 28);
    end generate gen_last_layer;

    o_layer_state <= layer_state;
    o_wrfull <= wrfull;
    o_q <= q_reg_reg;
    o_last <= last_reg_reg;
    o_rdempty <= rdempty_reg_reg;

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
            e_last_fifo : entity work.ip_dcfifo_mixed_widths
            generic map(
                ADDR_WIDTH_w    => TREE_w,
                DATA_WIDTH_w    => w_width,
                ADDR_WIDTH_r    => TREE_r,
                DATA_WIDTH_r    => r_width,
                DEVICE          => "Arria 10"--,
            )
            port map (
                aclr    => reset_fifo(i),
                data    => data(i),
                rdclk   => i_clk,
                rdreq   => rdreq(i),
                wrclk   => i_clk,
                wrreq   => wrreq(i),
                q       => last,
                rdempty => rdempty(i),
                wrfull  => wrfull(i)--,
            );

            -- reg for last FIFO output (timing)
            rdreq(i) <= '1' when rdempty(i) = '0' and wrfull_reg(i) = '0' else '0';
            rdreq_reg(i) <= '1' when rdempty_reg(i) = '0' and wrfull_reg_reg(i) = '0' else '0';
            process(i_clk, reset_fifo(i))
            begin
            if ( reset_fifo(i) = '1' ) then
                rdempty_reg(i)    <= '1';
                wrfull_reg(i)     <= '0';
                last_reg          <= (others => '0');
                rdempty_reg_reg(i)<= '1';
                wrfull_reg_reg(i) <= '0';
                last_reg_reg      <= (others => '0');
                --
            elsif ( rising_edge(i_clk) ) then

                if ( rdreq(i) = '1' ) then
                    last_reg       <= last;
                    wrfull_reg(i)  <= '1';
                    rdempty_reg(i) <= '0';
                end if;

                if ( rdreq_reg(i) = '1' ) then
                    last_reg_reg   <= last_reg;
                    wrfull_reg(i)  <= '0';
                    rdempty_reg(i) <= '1';

                    wrfull_reg_reg(i)  <= '1';
                    rdempty_reg_reg(i) <= '0';
                end if;

                if ( i_rdreq(i) = '1' ) then
                    wrfull_reg_reg(i)  <= '0';
                    rdempty_reg_reg(i) <= '1';
                end if;

            end if;
            end process;

        END GENERATE;

        gen_layer : IF last_layer = '0' and i < g_NLINKS_DATA GENERATE
            e_link_fifo : entity work.ip_dcfifo_mixed_widths
            generic map(
                ADDR_WIDTH_w    => TREE_w,
                DATA_WIDTH_w    => w_width,
                ADDR_WIDTH_r    => TREE_r,
                DATA_WIDTH_r    => r_width,
                DEVICE          => "Arria 10"--,
            )
            port map (
                aclr    => reset_fifo(i),
                data    => data(i),
                rdclk   => i_clk,
                rdreq   => rdreq(i),
                wrclk   => i_clk,
                wrreq   => wrreq(i),
                q       => q(i),
                rdempty => rdempty(i),
                wrfull  => wrfull(i)--,
            );

            -- reg for FIFO output (timing)
            rdreq(i) <= '1' when rdempty(i) = '0' and wrfull_reg(i) = '0' else '0';
            rdreq_reg(i) <= '1' when rdempty_reg(i) = '0' and wrfull_reg_reg(i) = '0' else '0';
            process(i_clk, reset_fifo(i))
            begin
            if ( reset_fifo(i) = '1' ) then
                rdempty_reg(i)    <= '1';
                wrfull_reg(i)     <= '0';
                q_reg(i)          <= (others => '0');
                rdempty_reg_reg(i)<= '1';
                wrfull_reg_reg(i) <= '0';
                q_reg_reg(i)       <= (others => '0');
            elsif ( rising_edge(i_clk) ) then

                if ( rdreq(i) = '1' ) then
                    q_reg(i)       <= q(i);
                    wrfull_reg(i)  <= '1';
                    rdempty_reg(i) <= '0';
                end if;

                if ( rdreq_reg(i) = '1' ) then
                    q_reg_reg(i)   <= q_reg(i);
                    wrfull_reg(i)  <= '0';
                    rdempty_reg(i) <= '1';

                    wrfull_reg_reg(i)  <= '1';
                    rdempty_reg_reg(i) <= '0';
                end if;

                if ( i_rdreq(i) = '1' ) then
                    wrfull_reg_reg(i)  <= '0';
                    rdempty_reg_reg(i) <= '1';
                end if;

            end if;
            end process;

        END GENERATE;


        -- Tree setup
        -- x => empty, F => padding
        -- [b,a]
        -- [1,1]  -> [1,1] -> [x,2] -> [3,2] -> [2,2] -> [x,3] -> [3,3] -> ->  [3,2] -> [2,2] [4,2] -> [4,2]
        -- [2,2]              [2,2]    [2,2]             [x,2]    [3,2]        [F,2]          [F,F]
        -- [d,c]

        -- TODO: what to do when the FIFO (wrfull(i) = '1') gets full? There need to be a waiting state which does nothing
        wrfull_and_merge_state(i)                               <= '1' when i_merge_state = '1' and wrfull(i) = '0' else '0';
        both_inputs_rdempty(i)                                  <= '0' when i_rdempty(i) = '0' and i_rdempty(i+size) = '0' else '1';
        wrfull_and_merge_state_and_both_inputs_not_rdempty(i)   <= '1' when wrfull_and_merge_state(i) = '1' and both_inputs_rdempty(i) = '0' else '0';
        wrfull_and_merge_state_and_first_input_not_rdempty(i)   <= '1' when wrfull_and_merge_state(i) = '1' and i_rdempty(i) = '0' else '0';
        wrfull_and_merge_state_and_second_input_not_rdempty(i)  <= '1' when wrfull_and_merge_state(i) = '1' and i_rdempty(i+size) = '0' else '0';
        first_input_mask_n_second_input_not_mask_n(i)           <= '1' when i_mask_n(i) = '1' and i_mask_n(i+size) = '0' else '0';
        second_input_mask_n_first_input_not_mask_n(i)           <= '1' when i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else '0';
        a_padding(i)        <= '1' when a_h(i) = tree_padding else '0';
        b_padding(i)        <= '1' when b_h(i) = tree_padding else '0';
        a_b_padding(i)      <= '1' when a_padding(i) = '1' and b_padding(i) = '1' else '0';
        a_no_b_padding(i)   <= '1' when a_padding(i) = '0' and b_padding(i) = '1' else '0';
        b_no_a_padding(i)   <= '1' when a_padding(i) = '1' and b_padding(i) = '0' else '0';

        -- TODO: name the different states, combine stuff
        -- TODO: include sub-header, check backpres., counters etc.
        layer_state(i) <= last_layer_state when i_merge_state = '0' and last_layer = '1' else

                          second_input_not_mask_n when wrfull_and_merge_state_and_first_input_not_rdempty(i) = '1' and first_input_mask_n_second_input_not_mask_n(i) = '1' else
                          first_input_not_mask_n when wrfull_and_merge_state_and_second_input_not_rdempty(i) = '1' and second_input_mask_n_first_input_not_mask_n(i) = '1' else

                          a_no_b_padding_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' else
                          b_no_a_padding_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b_no_a_padding(i) = '1' else

                          a_smaller_b when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= b(i) else
                          b_smaller_a when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i)  > b(i) else

                          end_state when a_b_padding(i) = '1' else

                          IDEL;

        wrreq(i)        <=  '1' when layer_state(i) = last_layer_state and i_wen_h_t = '1' else
                            '1' when layer_state(i) = end_state else
                            '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = b_no_a_padding_state or layer_state(i) = a_smaller_b or layer_state(i) = b_smaller_a else
                            '0';

        o_rdreq(i)      <=  '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = a_smaller_b else
                            '0';

        o_rdreq(i+size) <=  '1' when layer_state(i) = first_input_not_mask_n or layer_state(i) = b_no_a_padding_state or layer_state(i) = b_smaller_a else
                            '0';

        data(i)         <= i_data_h_t when layer_state(i) = last_layer_state else
                           tree_padding when layer_state(i) = end_state else
                           a_h(i) when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = a_smaller_b else
                           b_h(i) when layer_state(i) = first_input_not_mask_n or layer_state(i) = b_no_a_padding_state or layer_state(i) = b_smaller_a else
                           (others => '0');

    END GENERATE;

end architecture;
