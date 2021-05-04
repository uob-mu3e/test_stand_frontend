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
    o_rdempty       : out std_logic_vector(gen_fifos - 1 downto 0);
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
    constant end_state : std_logic_vector(7 downto 0) := x"08";
    constant last_layer_state : std_logic_vector(7 downto 0) := x"09";
    constant read_out_first_second_padding : std_logic_vector(7 downto 0) := x"0A";
    constant read_out_second_first_padding : std_logic_vector(7 downto 0) := x"0B";
    constant read_out_a_rest_padding : std_logic_vector(7 downto 0) := x"0C";
    constant read_out_c_rest_padding : std_logic_vector(7 downto 0) := x"0D";

    signal data, data_reg, f_data, f_data_reg, q, q_reg : work.util.slv76_array_t(gen_fifos - 1 downto 0);
    signal layer_state, layer_state_reg : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal wrreq, f_wrreq, f_wrreq_reg, wrfull, reset_fifo, wrfull_and_merge_state, both_inputs_rdempty, rdempty, rdempty_reg, wrfull_reg, rdreq : std_logic_vector(gen_fifos - 1 downto 0);
    signal wrfull_and_merge_state_and_both_inputs_not_rdempty, wrfull_and_merge_state_and_first_input_not_rdempty, wrfull_and_merge_state_and_second_input_not_rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal first_input_mask_n_second_input_not_mask_n, second_input_mask_n_first_input_not_mask_n, a_padding, b_padding, c_padding, d_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a_b_padding, a_b_no_padding, a_no_b_padding, c_d_padding, c_d_no_padding, c_no_d_padding, a_c_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a, b, c, d : work.util.slv4_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_h, b_h, c_h, d_h : work.util.slv38_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_z, b_z, c_z, d_z : std_logic_vector(gen_fifos - 1 downto 0) := (others => '0');
    signal last, last_reg : std_logic_vector(r_width-1 downto 0);

    -- for debugging / simulation
    signal t_q, t_data : work.util.slv8_array_t(gen_fifos - 1 downto 0);
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
        t_q(i)(7 downto 4) <= q(i)(69 downto 66) when last_layer = '0' else last(69 downto 66);
        t_q(i)(3 downto 0) <= q(i)(31 downto 28) when last_layer = '0' else last(31 downto 28);
        t_data(i)(7 downto 4) <= data(i)(69 downto 66);
        t_data(i)(3 downto 0) <= data(i)(31 downto 28);
        l1(i) <= q(i)(75 downto 70) when last_layer = '0' else last(75 downto 70);
        l2(i) <= q(i)(37 downto 32) when last_layer = '0' else last(37 downto 32);
    END GENERATE;
    
    o_layer_state <= layer_state;
    o_wrfull <= wrfull;
    o_q <= q_reg;
    o_last <= last_reg;
    o_rdempty <= rdempty_reg;

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
            reg : process(i_clk, reset_fifo(i))
            begin
            if ( reset_fifo(i) = '1' ) then
                last_reg       <= (others => '0');
                rdreq(i)       <= '0';
                wrfull_reg(i)  <= '0';
                rdempty_reg(i) <= '1';
            elsif ( rising_edge(i_clk) ) then
                rdreq(i) <= '0';
                if ( rdempty(i) = '0' and (wrfull_reg(i) = '0' or i_rdreq(i) = '1') ) then
                    rdreq(i)       <= '1';
                    last_reg    <= last;
                    wrfull_reg(i)  <= '1';
                end if;
                
                if ( i_rdreq(i) = '1' ) then
                    wrfull_reg(i)  <= '0';
                end if;
                
                if ( rdempty(i) = '0' and i_rdreq(i) = '1' ) then
                    rdempty_reg(i) <= '0';
                elsif ( rdempty(i) = '0' and wrfull_reg(i) = '0' ) then
                    rdempty_reg(i) <= '0';
                elsif ( i_rdreq(i) = '1' ) then
                    rdempty_reg(i) <= '1';
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
            
            -- reg for FIFO outputs (timing)
            reg : process(i_clk, reset_fifo(i))
            begin
            if ( reset_fifo(i) = '1' ) then
                q_reg(i)       <= (others => '0');
                rdreq(i)       <= '0';
                wrfull_reg(i)  <= '0';
                rdempty_reg(i) <= '1';
            elsif ( rising_edge(i_clk) ) then
                rdreq(i) <= '0';
                if ( rdempty(i) = '0' and (wrfull_reg(i) = '0' or i_rdreq(i) = '1') ) then
                    rdreq(i)       <= '1';
                    q_reg(i)       <= q(i);
                    wrfull_reg(i)  <= '1';
                end if;
                
                if ( i_rdreq(i) = '1' ) then
                    wrfull_reg(i)  <= '0';
                end if;
                
                if ( rdempty(i) = '0' and i_rdreq(i) = '1' ) then
                    rdempty_reg(i) <= '0';
                elsif ( rdempty(i) = '0' and wrfull_reg(i) = '0' ) then
                    rdempty_reg(i) <= '0';
                elsif ( i_rdreq(i) = '1' ) then
                    rdempty_reg(i) <= '1';
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

                          end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_c_padding(i) = '1' else
                          -- this should never be possible since checking empty is enough
                          --end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_h(i) = tree_padding and c_z(i) = '1' else
                          --end_state when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_z(i) = '1'          and c_h(i) = tree_padding else

                          second_input_not_mask_n when wrfull_and_merge_state_and_first_input_not_rdempty(i) = '1' and first_input_mask_n_second_input_not_mask_n(i) = '1' else
                          first_input_not_mask_n when wrfull_and_merge_state_and_second_input_not_rdempty(i) = '1' and second_input_mask_n_first_input_not_mask_n(i) = '1' else
                          
                          read_out_first_second_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_b_no_padding(i) = '1' and c_d_padding(i) = '1' else-- and a_z(i) = '0' and b_z(i) = '0' else
                          read_out_second_first_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c_d_no_padding(i) = '1' and a_b_padding(i) = '1' else-- and c_z(i) = '0' and d_z(i) = '0' else
                          
                          read_out_a_rest_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_d_padding(i) = '1' else-- and a_z(i) = '0' else
                          read_out_c_rest_padding when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c_no_d_padding(i) = '1' and a_b_padding(i) = '1' else-- and c_z(i) = '0' else
                          
                          x"0E" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) <= c(i) else-- and a_z(i) = '0' and c_z(i) = '0' else
                          x"10" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) >  c(i) else-- and a_z(i) = '0' and c_z(i) = '0' else

                          x"11" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_d_no_padding(i) = '1' and a(i) <= c(i) else-- and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else                          
                          x"12" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_b_no_padding(i) = '1' and c_no_d_padding(i) = '1' and c(i) <= a(i) else-- and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          x"13" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_no_b_padding(i) = '1' and c_d_no_padding(i) = '1' and c(i) <= a(i) and a(i) <= d(i) else-- and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else                          
                          x"14" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a_b_no_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) <= c(i) and c(i) <= b(i) else-- and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          x"15" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) /= x"06" and a_no_b_padding(i) = '1' and c_d_no_padding(i) = '1' and c(i) <= a(i) and d(i) <= a(i) else-- and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else
                          x"16" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and layer_state_reg(i) /= x"07" and a_b_no_padding(i) = '1' and c_no_d_padding(i) = '1' and a(i) <= c(i) and b(i) <= c(i) else-- and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          
                          x"06" when (layer_state_reg(i) = x"04" or layer_state_reg(i) = x"05") and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) <= d(i) else-- and b_z(i) = '0' and d_z(i) = '0' else
                          x"07" when (layer_state_reg(i) = x"04" or layer_state_reg(i) = x"05") and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) >  d(i) else-- and b_z(i) = '0' and d_z(i) = '0' else
                          
                          x"04" when layer_state_reg(i) = x"06" and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= d(i) else-- and a_z(i) = '0' and d_z(i) = '0' else
                          x"0F" when layer_state_reg(i) = x"06" and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) >  d(i) else-- and a_z(i) = '0' and d_z(i) = '0' else
                          
                          x"0F" when layer_state_reg(i) = x"07" and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) <= c(i) else-- and b_z(i) = '0' and c_z(i) = '0' else
                          x"04" when layer_state_reg(i) = x"07" and wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and b(i) >  c(i) else-- and b_z(i) = '0' and c_z(i) = '0' else
                          
                          x"04" when layer_state_reg(i) = x"04" else
                          x"05" when layer_state_reg(i) = x"05" else
                          x"06" when layer_state_reg(i) = x"06" else
                          x"07" when layer_state_reg(i) = x"07" else
                          
                          x"02" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= c(i) and b(i) <= c(i) else-- and a_z(i) = '0' and b_z(i) = '0' else
                          x"03" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c(i) <= a(i) and d(i) <= a(i) else-- and c_z(i) = '0' and d_z(i) = '0' else  
                          x"04" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and a(i) <= c(i) and b(i) >  c(i) else-- and a_z(i) = '0' and c_z(i) = '0' else
                          x"05" when wrfull_and_merge_state_and_both_inputs_not_rdempty(i) = '1' and c(i) <= a(i) and d(i) >  a(i) else-- and c_z(i) = '0' and a_z(i) = '0' else
                          
                          x"0F";
                         
        wrreq(i) <= '1' when layer_state(i) = last_layer_state and i_wen_h_t = '1' else
                    '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = first_input_not_mask_n or layer_state(i) = x"02" or layer_state(i) = x"03" or layer_state(i) = x"04" or layer_state(i) = x"05" or layer_state(i) = x"08" or layer_state(i) = read_out_first_second_padding or layer_state(i) = read_out_second_first_padding or layer_state(i) = read_out_a_rest_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = x"0E" or layer_state(i) = x"10" or layer_state(i) = x"13" or layer_state(i) = x"14" or layer_state(i) = x"15" or layer_state(i) = x"16" else
                    '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else
                    '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else
                    '1' when layer_state(i) = x"04" and layer_state_reg(i) = x"06" else
                    '1' when layer_state(i) = x"04" and layer_state_reg(i) = x"07" else
                    '1' when layer_state(i) = read_out_a_rest_padding and layer_state_reg(i) = x"11" else
                    '1' when layer_state(i) = read_out_a_rest_padding and layer_state_reg(i) = x"12" else
                    '1' when layer_state(i) = read_out_c_rest_padding and layer_state_reg(i) = x"11" else
                    '1' when layer_state(i) = read_out_c_rest_padding and layer_state_reg(i) = x"11" else
                    '0';
                    
        o_rdreq(i) <= '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = x"02" or layer_state(i) = x"06" or layer_state(i) = read_out_first_second_padding or layer_state(i) = read_out_a_rest_padding or layer_state(i) = x"0E" or layer_state(i) = x"10" or layer_state(i) = x"11" or layer_state(i) = x"13" or layer_state(i) = x"16" else
                      '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else
                      '1' when layer_state(i) = x"06" and layer_state_reg(i) = x"04" else
                      '1' when layer_state(i) = x"06" and layer_state_reg(i) = x"05" else
                      '0';

        o_rdreq(i+size) <=  '1' when layer_state(i) = first_input_not_mask_n or layer_state(i) = x"03" or layer_state(i) = x"07" or layer_state(i) = read_out_second_first_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = x"0E" or layer_state(i) = x"10" or layer_state(i) = x"12" or layer_state(i) = x"14" or layer_state(i) = x"15" else
                            '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else
                            '1' when layer_state(i) = x"07" and layer_state_reg(i) = x"04" else
                            '1' when layer_state(i) = x"07" and layer_state_reg(i) = x"05" else
                            '0';
        
        -- TODO: combine logic of data_reg(i)(37 downto 0)
        data(i)(37 downto 0) <= i_data_h_t when layer_state(i) = last_layer_state else
                                tree_padding when layer_state(i) = end_state else
                                b_h(i) when layer_state(i) = x"06" and layer_state_reg(i) = x"04" else
                                b_h(i) when layer_state(i) = x"06" and layer_state_reg(i) = x"05" else
                                d_h(i) when layer_state(i) = x"07" and layer_state_reg(i) = x"04" else
                                d_h(i) when layer_state(i) = x"07" and layer_state_reg(i) = x"05" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"06" and layer_state_reg(i) = x"06" else 
                                data_reg(i)(37 downto 0) when layer_state(i) = x"07" and layer_state_reg(i) = x"07" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"04" and layer_state_reg(i) = x"06" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"04" and layer_state_reg(i) = x"07" else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_out_a_rest_padding and layer_state_reg(i) = x"11" else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_out_a_rest_padding and layer_state_reg(i) = x"12" else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_out_c_rest_padding and layer_state_reg(i) = x"11" else
                                data_reg(i)(37 downto 0) when layer_state(i) = read_out_c_rest_padding and layer_state_reg(i) = x"12" else
                                a_h(i) when layer_state(i) = second_input_not_mask_n or layer_state(i) = x"02" or layer_state(i) = x"04" or layer_state(i) = read_out_first_second_padding or layer_state(i) = read_out_a_rest_padding or layer_state(i) = x"0E" or layer_state(i) = x"11" or layer_state(i) = x"14" or layer_state(i) = x"16" else
                                c_h(i) when layer_state(i) = first_input_not_mask_n or layer_state(i) = x"03" or layer_state(i) = x"05" or layer_state(i) = read_out_second_first_padding or layer_state(i) = read_out_c_rest_padding or layer_state(i) = x"10" or layer_state(i) = x"12" or layer_state(i) = x"13" or layer_state(i) = x"15" else
                                b_h(i) when layer_state(i) = x"06" else
                                d_h(i) when layer_state(i) = x"07" else
                                (others => '0');

        data(i)(75 downto 38)<= tree_paddingk when layer_state(i) = last_layer_state else
                                tree_padding when layer_state(i) = end_state or layer_state(i) = read_out_a_rest_padding or layer_state(i) = read_out_c_rest_padding else
                                data_reg(i)(75 downto 38) when layer_state(i) = x"06" and layer_state_reg(i) = x"06" else
                                data_reg(i)(75 downto 38) when layer_state(i) = x"07" and layer_state_reg(i) = x"07" else
                                a_h(i) when layer_state(i) = read_out_a_rest_padding and layer_state_reg(i) = x"11" else
                                a_h(i) when layer_state(i) = read_out_a_rest_padding and layer_state_reg(i) = x"12" else
                                c_h(i) when layer_state(i) = read_out_c_rest_padding and layer_state_reg(i) = x"11" else
                                c_h(i) when layer_state(i) = read_out_c_rest_padding and layer_state_reg(i) = x"12" else
                                d_h(i) when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else 
                                b_h(i) when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else 
                                a_h(i) when layer_state(i) = x"04" and layer_state_reg(i) = x"06" else
                                c_h(i) when layer_state(i) = x"04" and layer_state_reg(i) = x"07" else
                                b_h(i) when layer_state(i) = second_input_not_mask_n or layer_state(i) = x"02" or layer_state(i) = read_out_first_second_padding or layer_state(i) = x"16" else
                                d_h(i) when layer_state(i) = x"15" or layer_state(i) = first_input_not_mask_n or layer_state(i) = x"03" or layer_state(i) = read_out_second_first_padding else
                                a_h(i) when layer_state(i) = x"05" or layer_state(i) = x"10" or layer_state(i) = x"13" else
                                c_h(i) when layer_state(i) = x"04" or layer_state(i) = x"0E" or layer_state(i) = x"14" else
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
