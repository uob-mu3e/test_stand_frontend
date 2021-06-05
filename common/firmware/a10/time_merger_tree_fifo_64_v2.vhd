library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


entity time_merger_tree_fifo_64_v2 is
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
    i_data          : in  work.util.slv76_array_t(compare_fifos - 1 downto 0);
    i_rdempty       : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_rdreq         : in  std_logic_vector(gen_fifos - 1 downto 0);
    i_merge_state   : in  std_logic;
    i_mask_n        : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_wen_h_t       : in  std_logic;
    i_data_h_t      : in  std_logic_vector(37 downto 0);

    -- output
    o_q             : out work.util.slv76_array_t(gen_fifos - 1 downto 0);
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

architecture arch of time_merger_tree_fifo_64_v2 is

    -- merger signals
    constant size : integer := compare_fifos/2;
    
    -- layer states
    constant second_input_not_mask_n : std_logic_vector(7 downto 0) := x"00";
    constant first_input_not_mask_n : std_logic_vector(7 downto 0) := x"01";
    constant a_b_smaller_c_d : std_logic_vector(7 downto 0) := x"02";
    constant c_d_smaller_a_b : std_logic_vector(7 downto 0) := x"03";
    constant a_smaller_c_c_smaller_b : std_logic_vector(7 downto 0) := x"04";
    constant c_smaller_a_a_smaller_d : std_logic_vector(7 downto 0) := x"05";
    constant b_smaller_d : std_logic_vector(7 downto 0) := x"06";
    constant d_smaller_b : std_logic_vector(7 downto 0) := x"07";
    constant end_state : std_logic_vector(7 downto 0) := x"08";
    constant last_layer_state : std_logic_vector(7 downto 0) := x"09";
    constant read_out_first_second_padding : std_logic_vector(7 downto 0) := x"0A";
    constant read_out_second_first_padding : std_logic_vector(7 downto 0) := x"0B";
    constant read_out_a_rest_padding : std_logic_vector(7 downto 0) := x"0C";
    constant read_out_c_rest_padding : std_logic_vector(7 downto 0) := x"0D";
    constant d_smaller_a : std_logic_vector(7 downto 0) := x"0E";
    constant b_smaller_c : std_logic_vector(7 downto 0) := x"0F";
    constant a_smaller_c_rest_padding : std_logic_vector(7 downto 0) := x"10";
    constant c_smaller_a_rest_padding : std_logic_vector(7 downto 0) := x"11";
    constant a_smaller_c_b_padding : std_logic_vector(7 downto 0) := x"12";
    constant read_d_rest_padding : std_logic_vector(7 downto 0) := x"13";
    constant write_d_set_padding : std_logic_vector(7 downto 0) := x"14";
    constant write_d_c : std_logic_vector(7 downto 0) := x"15";
    constant c_smaller_a_d_padding : std_logic_vector(7 downto 0) := x"16";
    constant read_b_rest_padding : std_logic_vector(7 downto 0) := x"17";
    constant write_b_set_padding : std_logic_vector(7 downto 0) := x"18";
    constant write_b_a : std_logic_vector(7 downto 0) := x"19";
    constant c_smaller_a_b_padding : std_logic_vector(7 downto 0) := x"1A";
    constant a_smaller_c_d_padding : std_logic_vector(7 downto 0) := x"1B";
    constant c_d_smaller_a_b_padding : std_logic_vector(7 downto 0) := x"1C";
    constant a_b_smaller_c_d_padding : std_logic_vector(7 downto 0) := x"1D";
    constant read_out_b_and_c : std_logic_vector(7 downto 0) := x"1E";
    constant read_out_d_and_a : std_logic_vector(7 downto 0) := x"1F";
    constant IDEL : std_logic_vector(7 downto 0) := x"FF";

    signal data, data_reg, f_data, f_data_reg, q, q_reg, q_reg_reg : work.util.slv76_array_t(gen_fifos - 1 downto 0);
    signal layer_state, layer_state_reg : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal wrreq, f_wrreq, f_wrreq_reg, wrfull, reset_fifo, wrfull_and_merge_state, both_inputs_rdempty, rdempty, rdempty_reg, wrfull_reg, rdreq : std_logic_vector(gen_fifos - 1 downto 0);
    signal rdempty_reg_reg, wrfull_reg_reg, rdreq_reg : std_logic_vector(gen_fifos - 1 downto 0);
    signal wrfull_and_merge_state_and_both_inputs_not_rdempty, wrfull_and_merge_state_and_first_input_not_rdempty, wrfull_and_merge_state_and_second_input_not_rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal first_input_mask_n_second_input_not_mask_n, second_input_mask_n_first_input_not_mask_n, a_padding, b_padding, c_padding, d_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a_b_padding, a_b_no_padding, a_no_b_padding, c_d_padding, c_d_no_padding, c_no_d_padding, a_c_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a, b, c, d : work.util.slv4_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_h, b_h, c_h, d_h : work.util.slv38_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_z, b_z, c_z, d_z : std_logic_vector(gen_fifos - 1 downto 0) := (others => '0');
    signal last, last_reg, last_reg_reg : std_logic_vector(r_width-1 downto 0);

    -- for debugging / simulation
    signal t_q, t_data : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal t_q_last : std_logic_vector(31 downto 0);
    signal l1 : work.util.slv6_array_t(gen_fifos - 1 downto 0);
    signal l2 : work.util.slv6_array_t(gen_fifos - 1 downto 0);

begin
    
    gen_hits:
    FOR i in 0 to gen_fifos - 1 GENERATE
        a(i) <= i_data(i)(31 downto 28);
        b(i) <= i_data(i)(69 downto 66);
        c(i) <= i_data(i+size)(31 downto 28);
        d(i) <= i_data(i+size)(69 downto 66);
        
        a_h(i) <= i_data(i)(37 downto 0);
        b_h(i) <= i_data(i)(75 downto 38);
        c_h(i) <= i_data(i+size)(37 downto 0);
        d_h(i) <= i_data(i+size)(75 downto 38);

        -- TODO: set tree_padding == x"FFFFFFFF"
        a_z(i) <= '1' when a_h(i) = tree_zero else '0';
        b_z(i) <= '1' when b_h(i) = tree_zero else '0';
        c_z(i) <= '1' when c_h(i) = tree_zero else '0';
        d_z(i) <= '1' when d_h(i) = tree_zero else '0';
        
        -- for debugging / simulation
        t_q(i)(7 downto 4) <= q_reg_reg(i)(69 downto 66);
        t_q(i)(3 downto 0) <= q_reg_reg(i)(31 downto 28);
        t_data(i)(7 downto 4) <= data(i)(69 downto 66);
        t_data(i)(3 downto 0) <= data(i)(31 downto 28);
        l1(i) <= q_reg_reg(i)(75 downto 70) when last_layer = '0' else last_reg_reg(75 downto 70);
        l2(i) <= q_reg_reg(i)(37 downto 32) when last_layer = '0' else last_reg_reg(37 downto 32);
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
--        reset_fifo(i) <= '0' when i_merge_state = '1' or last_layer = '1' else '1';
        
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
        
        wrfull_and_merge_state(i)                               <= '1' when i_merge_state = '1' and wrfull(i) = '0' else '0';
        both_inputs_rdempty(i)                                  <= '0' when i_rdempty(i) = '0' and i_rdempty(i+size) = '0' else '1';
        wrfull_and_merge_state_and_both_inputs_not_rdempty(i)   <= '1' when wrfull_and_merge_state(i) = '1' and both_inputs_rdempty(i) = '0' else '0';
        wrfull_and_merge_state_and_first_input_not_rdempty(i)   <= '1' when wrfull_and_merge_state(i) = '1' and i_rdempty(i) = '0' else '0';
        wrfull_and_merge_state_and_second_input_not_rdempty(i)  <= '1' when wrfull_and_merge_state(i) = '1' and i_rdempty(i+size) = '0' else '0';
        first_input_mask_n_second_input_not_mask_n(i)           <= '1' when i_mask_n(i) = '1' and i_mask_n(i+size) = '0' else '0';
        second_input_mask_n_first_input_not_mask_n(i)           <= '1' when i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else '0';
        a_padding(i)        <= '1' when a_h(i) = tree_padding else '0';
        b_padding(i)        <= '1' when b_h(i) = tree_padding else '0'; 
        c_padding(i)        <= '1' when c_h(i) = tree_padding else '0';
        d_padding(i)        <= '1' when d_h(i) = tree_padding else '0';
        a_b_padding(i)      <= '1' when a_padding(i) = '1' and b_padding(i) = '1' else '0';
        a_b_no_padding(i)   <= '1' when a_padding(i) = '0' and b_padding(i) = '0' else '0';
        a_no_b_padding(i)   <= '1' when a_padding(i) = '0' and b_padding(i) = '1' else '0';
        c_d_padding(i)      <= '1' when c_padding(i) = '1' and d_padding(i) = '1' else '0';
        c_d_no_padding(i)   <= '1' when c_padding(i) = '0' and d_padding(i) = '0' else '0';
        c_no_d_padding(i)   <= '1' when c_padding(i) = '0' and d_padding(i) = '1' else '0';
        a_c_padding(i)      <= '1' when a_padding(i) = '1' and c_padding(i) = '1' else '0';
        
        -- TODO: name the different states, combine stuff
        -- TODO: include sub-header, check backpres., counters etc.
        layer_state(i) <= last_layer_state when i_merge_state = '0' and last_layer = '1' else
        
                          write_d_set_padding when wrfull_and_merge_state(i) = '1' and layer_state_reg(i) = read_d_rest_padding and a_c_padding(i) = '1' else
                          write_d_c when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) = read_d_rest_padding else
                          read_d_rest_padding when layer_state_reg(i) = read_d_rest_padding or layer_state_reg(i) = read_out_b_and_c else
                          
                          write_b_set_padding when wrfull_and_merge_state(i) = '1' and layer_state_reg(i) = read_b_rest_padding and a_c_padding(i) = '1' else
                          write_b_a when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) = read_b_rest_padding else
                          read_b_rest_padding when layer_state_reg(i) = read_b_rest_padding or layer_state_reg(i) = read_out_d_and_a else

                          end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_c_padding(i) = '1' else
                          end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) = write_d_c and d_padding(i) = '1' else
                          -- this should never be possible since checking empty is enough
                          --end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_h(i) = tree_padding and c_z(i) = '1' else
                          --end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_z(i) = '1'          and c_h(i) = tree_padding else
                          read_d_rest_padding when layer_state_reg(i) = a_smaller_c_b_padding or layer_state_reg(i) = c_smaller_a_b_padding or (layer_state_reg(i) = write_d_c and d_padding(i) = '0') else
                          -- TODO: check this a_smaller_c_d_padding
                          read_b_rest_padding when layer_state_reg(i) = c_smaller_a_d_padding or (layer_state_reg(i) = write_b_a and b_padding(i) = '0') else

                          second_input_not_mask_n when wrfull_and_merge_state_and_first_input_not_rdempty(i) = '1' and first_input_mask_n_second_input_not_mask_n(i) = '1' else
                          first_input_not_mask_n when wrfull_and_merge_state_and_second_input_not_rdempty(i) = '1' and second_input_mask_n_first_input_not_mask_n(i) = '1' else

                          read_out_b_and_c when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) = b_smaller_d and c_d_no_padding(i) = '1' and a_b_padding(i) = '1' else
                          read_out_d_and_a when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) = d_smaller_b and a_b_no_padding(i) = '1' and c_d_padding(i) = '1' else
                          
                          read_out_first_second_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_b_no_padding(i) = '1' and c_d_padding(i) = '1' else-- and a_z(i) = '0' and b_z(i) = '0' else
                          read_out_second_first_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c_d_no_padding(i) = '1' and a_b_padding(i) = '1' else-- and c_z(i) = '0' and d_z(i) = '0' else
                          
                          read_out_a_rest_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_d_padding(i) = '1' else-- and a_z(i) = '0' else
                          read_out_c_rest_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c_no_d_padding(i) = '1' and a_b_padding(i) = '1' else-- and c_z(i) = '0' else
                          
                          a_smaller_c_rest_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) <= c(i) else-- and a_z(i) = '0' and c_z(i) = '0' else
                          c_smaller_a_rest_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) >  c(i) else-- and a_z(i) = '0' and c_z(i) = '0' else

                          a_smaller_c_b_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_d_no_padding(i) = '1' and a(i) <= c(i) else-- and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else                          
                          c_smaller_a_d_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_b_no_padding(i) = '1' and c_no_d_padding(i) = '1' and c(i) <= a(i) else-- and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          c_smaller_a_b_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_d_no_padding(i) = '1' and c(i) <= a(i) and a(i) <= d(i) else-- and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else                          
                          a_smaller_c_d_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_b_no_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) <= c(i) and c(i) <= b(i) else-- and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          c_d_smaller_a_b_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) /= b_smaller_d and a_no_b_padding(i) = '1' and c_d_no_padding(i) = '1' and c(i) <= a(i) and d(i) <= a(i) else-- and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else
                          a_b_smaller_c_d_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) /= d_smaller_b and a_b_no_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) <= c(i) and b(i) <= c(i) else-- and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          
                          b_smaller_d when (layer_state_reg(i) = a_smaller_c_c_smaller_b or layer_state_reg(i) = c_smaller_a_a_smaller_d) and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) <= d(i) else-- and b_z(i) = '0' and d_z(i) = '0' else
                          d_smaller_b when (layer_state_reg(i) = a_smaller_c_c_smaller_b or layer_state_reg(i) = c_smaller_a_a_smaller_d) and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) >  d(i) else-- and b_z(i) = '0' and d_z(i) = '0' else
                          
                          a_smaller_c_c_smaller_b when layer_state_reg(i) = b_smaller_d and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= d(i) else-- and a_z(i) = '0' and d_z(i) = '0' else
                          d_smaller_a when layer_state_reg(i) = b_smaller_d and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) >  d(i) else-- and a_z(i) = '0' and d_z(i) = '0' else
                          
                          a_smaller_c_c_smaller_b when layer_state_reg(i) = d_smaller_b and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) >  c(i) else-- and b_z(i) = '0' and c_z(i) = '0' else
                          b_smaller_c when layer_state_reg(i) = d_smaller_b and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) <= c(i) else-- and b_z(i) = '0' and c_z(i) = '0' else
                          
                          a_smaller_c_c_smaller_b when layer_state_reg(i) = a_smaller_c_c_smaller_b else
                          c_smaller_a_a_smaller_d when layer_state_reg(i) = c_smaller_a_a_smaller_d else
                          b_smaller_d when layer_state_reg(i) = b_smaller_d else
                          d_smaller_b when layer_state_reg(i) = d_smaller_b else
                          
                          a_b_smaller_c_d when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= c(i) and b(i) <= c(i) else-- and a_z(i) = '0' and b_z(i) = '0' else
                          c_d_smaller_a_b when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c(i) <= a(i) and d(i) <= a(i) else-- and c_z(i) = '0' and d_z(i) = '0' else  
                          a_smaller_c_c_smaller_b when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= c(i) and b(i) >  c(i) else-- and a_z(i) = '0' and c_z(i) = '0' else
                          c_smaller_a_a_smaller_d when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c(i) <= a(i) and d(i) >  a(i) else-- and c_z(i) = '0' and a_z(i) = '0' else
                          
                          IDEL;
                         
        wrreq(i) <= '1' when layer_state(i) = last_layer_state and i_wen_h_t = '1' else
                    '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = first_input_not_mask_n or layer_state(i) = a_b_smaller_c_d or layer_state(i) = c_d_smaller_a_b or layer_state(i) = a_smaller_c_c_smaller_b or layer_state(i) = c_smaller_a_a_smaller_d or layer_state(i) = end_state or layer_state(i) = read_out_first_second_padding or layer_state(i) = read_out_second_first_padding or layer_state(i) = read_out_a_rest_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = a_smaller_c_rest_padding or layer_state(i) = c_smaller_a_rest_padding or layer_state(i) = c_smaller_a_b_padding or layer_state(i) = a_smaller_c_d_padding or layer_state(i) = c_d_smaller_a_b_padding or layer_state(i) = a_b_smaller_c_d_padding or layer_state(i) = a_smaller_c_b_padding or layer_state(i) = write_d_set_padding or layer_state(i) = write_b_set_padding or layer_state(i) = write_d_c or layer_state(i) = c_smaller_a_d_padding or layer_state(i) = write_b_a or layer_state(i) = read_out_b_and_c or layer_state(i) = read_out_d_and_a else
                    '1' when layer_state(i) = d_smaller_a else
                    '1' when layer_state(i) = b_smaller_c else
                    --'1' when layer_state(i) = a_smaller_c_c_smaller_b and layer_state_reg(i) = b_smaller_d else --NOTE: maybe not needed
                    --'1' when layer_state(i) = a_smaller_c_c_smaller_b and layer_state_reg(i) = d_smaller_b else --NOTE: maybe not needed
                    '0';
                    
        o_rdreq(i) <= '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_b_smaller_c_d or (layer_state(i) = b_smaller_d and layer_state_reg(i) /= b_smaller_d) or layer_state(i) = read_out_first_second_padding or layer_state(i) = read_out_a_rest_padding or layer_state(i) = a_smaller_c_rest_padding or layer_state(i) = c_smaller_a_rest_padding or layer_state(i) = c_smaller_a_b_padding or layer_state(i) = a_b_smaller_c_d_padding or layer_state(i) = a_smaller_c_b_padding or layer_state(i) = read_b_rest_padding else
                      '1' when layer_state(i) = b_smaller_c else
                      '1' when layer_state(i) = b_smaller_d and layer_state_reg(i) = a_smaller_c_c_smaller_b else
                      '1' when layer_state(i) = b_smaller_d and layer_state_reg(i) = c_smaller_a_a_smaller_d else
                      '0';

        o_rdreq(i+size) <=  '1' when layer_state(i) = first_input_not_mask_n or layer_state(i) = c_d_smaller_a_b or (layer_state(i) = d_smaller_b and layer_state(i) /= d_smaller_b) or layer_state(i) = read_out_second_first_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = a_smaller_c_rest_padding or layer_state(i) = c_smaller_a_rest_padding or layer_state(i) = a_smaller_c_d_padding or layer_state(i) = c_d_smaller_a_b_padding or layer_state(i) = read_d_rest_padding or layer_state(i) = c_smaller_a_d_padding else
                            '1' when layer_state(i) = d_smaller_a else
                            '1' when layer_state(i) = d_smaller_b and layer_state_reg(i) = a_smaller_c_c_smaller_b else
                            '1' when layer_state(i) = d_smaller_b and layer_state_reg(i) = c_smaller_a_a_smaller_d else
                            '0';
        
        -- TODO: combine logic of data_reg(i)(37 downto 0)
        data(i)(37 downto 0) <= i_data_h_t when layer_state(i) = last_layer_state else
                                tree_padding when layer_state(i) = end_state else
                                b_h(i) when layer_state(i) = b_smaller_d and layer_state_reg(i) = a_smaller_c_c_smaller_b else
                                b_h(i) when layer_state(i) = b_smaller_d and layer_state_reg(i) = c_smaller_a_a_smaller_d else
                                d_h(i) when layer_state(i) = d_smaller_b and layer_state_reg(i) = a_smaller_c_c_smaller_b else
                                d_h(i) when layer_state(i) = d_smaller_b and layer_state_reg(i) = c_smaller_a_a_smaller_d else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_d_rest_padding and layer_state_reg(i) = read_d_rest_padding else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_b_rest_padding and layer_state_reg(i) = read_b_rest_padding else
                                d_h(i) when layer_state(i) = read_d_rest_padding else
                                b_h(i) when layer_state(i) = read_b_rest_padding else
                                data_reg(i)(37 downto 0) when layer_state(i) = write_d_set_padding else
                                data_reg(i)(37 downto 0) when layer_state(i) = write_b_set_padding else
                                data_reg(i)(37 downto 0) when layer_state(i) = write_d_c else
                                data_reg(i)(37 downto 0) when layer_state(i) = write_b_a else
                                data_reg(i)(37 downto 0) when layer_state(i) = b_smaller_d and layer_state_reg(i) = b_smaller_d else 
                                data_reg(i)(37 downto 0) when layer_state(i) = d_smaller_b and layer_state_reg(i) = d_smaller_b else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_out_b_and_c and layer_state_reg(i) = b_smaller_d else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_out_d_and_a and layer_state_reg(i) = d_smaller_b else
                                data_reg(i)(37 downto 0) when layer_state(i) = d_smaller_a else
                                data_reg(i)(37 downto 0) when layer_state(i) = b_smaller_c else
                                data_reg(i)(37 downto 0) when layer_state(i) = a_smaller_c_c_smaller_b and layer_state_reg(i) = b_smaller_d else
                                data_reg(i)(37 downto 0) when layer_state(i) = a_smaller_c_c_smaller_b and layer_state_reg(i) = d_smaller_b else
                                a_h(i) when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_b_smaller_c_d or layer_state(i) = a_smaller_c_c_smaller_b or layer_state(i) = read_out_first_second_padding or layer_state(i) = read_out_a_rest_padding or layer_state(i) = a_smaller_c_rest_padding or layer_state(i) = a_smaller_c_d_padding or layer_state(i) = a_b_smaller_c_d_padding or layer_state(i) = a_smaller_c_b_padding else
                                c_h(i) when layer_state(i) = first_input_not_mask_n or layer_state(i) = c_d_smaller_a_b or layer_state(i) = c_smaller_a_a_smaller_d or layer_state(i) = read_out_second_first_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = c_smaller_a_rest_padding or layer_state(i) = c_smaller_a_b_padding or layer_state(i) = c_d_smaller_a_b_padding or layer_state(i) = c_smaller_a_d_padding else
                                --b_h(i) when layer_state(i) = b_smaller_d else --NOTE: maybe not needed
                                --d_h(i) when layer_state(i) = d_smaller_b else --NOTE: maybe not needed
                                (others => '0');

        data(i)(75 downto 38)<= tree_paddingk when layer_state(i) = last_layer_state else
                                tree_padding when layer_state(i) = end_state or layer_state(i) = read_out_a_rest_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = write_d_set_padding or layer_state(i) = write_b_set_padding else
                                --data_reg(i)(75 downto 38) when layer_state(i) = b_smaller_d and layer_state_reg(i) = b_smaller_d else --NOTE: not needed
                                --data_reg(i)(75 downto 38) when layer_state(i) = d_smaller_b and layer_state_reg(i) = d_smaller_b else --NOTE: not needed
                                c_h(i) when layer_state(i) = read_out_b_and_c and layer_state_reg(i) = b_smaller_d else
                                a_h(i) when layer_state(i) = read_out_d_and_a and layer_state_reg(i) = d_smaller_b else
                                c_h(i) when layer_state(i) = a_smaller_c_b_padding else
                                a_h(i) when layer_state(i) = c_smaller_a_d_padding else
                                c_h(i) when layer_state(i) = write_d_c else
                                a_h(i) when layer_state(i) = write_b_a else
                                d_h(i) when layer_state(i) = d_smaller_a else 
                                b_h(i) when layer_state(i) = b_smaller_c else 
                                a_h(i) when layer_state(i) = a_smaller_c_c_smaller_b and layer_state_reg(i) = b_smaller_d else
                                c_h(i) when layer_state(i) = a_smaller_c_c_smaller_b and layer_state_reg(i) = d_smaller_b else
                                b_h(i) when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_b_smaller_c_d or layer_state(i) = read_out_first_second_padding or layer_state(i) = a_b_smaller_c_d_padding else
                                d_h(i) when layer_state(i) = c_d_smaller_a_b_padding or layer_state(i) = first_input_not_mask_n or layer_state(i) = c_d_smaller_a_b or layer_state(i) = read_out_second_first_padding else
                                a_h(i) when layer_state(i) = c_smaller_a_a_smaller_d or layer_state(i) = c_smaller_a_rest_padding or layer_state(i) = c_smaller_a_b_padding else
                                c_h(i) when layer_state(i) = a_smaller_c_c_smaller_b or layer_state(i) = a_smaller_c_rest_padding or layer_state(i) = a_smaller_c_d_padding else
                                (others => '0');
        
        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n /= '1' ) then
            layer_state_reg(i)  <= (others => '0');
            data_reg(i)         <= (others => '0');
            --
        elsif ( rising_edge(i_clk) ) then
            layer_state_reg(i)  <= layer_state(i);
            data_reg(i)         <= data(i);
        end if;
        end process;
        
        -- reg for FIFO inputs (timing)
--        process(i_clk, i_reset_n)
--        begin
--        if ( i_reset_n /= '1' ) then
--            f_data(i)       <= (others => '0');
--            f_wrreq(i)      <= '0';
--            f_data_reg(i)   <= (others => '0');
--            f_wrreq_reg(i)  <= '0';
--            --
--        elsif ( rising_edge(i_clk) ) then
--            f_data_reg(i)   <= data(i);
--            f_wrreq_reg(i)  <= wrreq(i);
--            f_data(i)       <= f_data_reg(i);
--            f_wrreq(i)      <= f_wrreq_reg(i);
--        end if;
--        end process;
        
        -- reg for FIFO outputs (timing)
--        process(i_clk, i_reset_n)
--        begin
--        if ( i_reset_n /= '1' ) then
--            q_reg(i)        <= (others => '0');
--            rdempty_reg(i)  <= '1';
--            --
--        elsif ( rising_edge(i_clk) ) then
--            q_reg(i)        <= q(i);
--            rdempty_reg(i)  <= rdempty(i);
--        end if;
--        end process;        
        
    END GENERATE;

end architecture;
