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
    
    signal data, data_reg, f_data, q : work.util.slv76_array_t(gen_fifos - 1 downto 0);
    signal layer_state, layer_state_reg : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal wrreq, f_wrreq, wrfull, reset_fifo, wrreq_good, both_rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal a, b, c, d : work.util.slv4_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_h, b_h, c_h, d_h : work.util.slv38_array_t(gen_fifos - 1 downto 0) := (others => (others => '0'));
    signal a_z, b_z, c_z, d_z : std_logic_vector(gen_fifos - 1 downto 0) := (others => '0');
    signal last :  std_logic_vector(r_width-1 downto 0);
    
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

        a_z(i) <= '1' when a_h(i) = tree_zero else '0';
        b_z(i) <= '1' when b_h(i) = tree_zero else '0';
        c_z(i) <= '1' when c_h(i) = tree_zero else '0';
        d_z(i) <= '1' when d_h(i) = tree_zero else '0';
        
        -- for debugging / simulation
        t_q(i)(7 downto 4) <= q(i)(69 downto 66) when last_layer = '0' else last(69 downto 66);
        t_q(i)(3 downto 0) <= q(i)(31 downto 28) when last_layer = '0' else last(31 downto 28);
        t_data(i)(7 downto 4) <= f_data(i)(69 downto 66);
        t_data(i)(3 downto 0) <= f_data(i)(31 downto 28);
        l1(i) <= q(i)(75 downto 70) when last_layer = '0' else last(75 downto 70);
        l2(i) <= q(i)(37 downto 32) when last_layer = '0' else last(37 downto 32);
    END GENERATE;
    
    o_layer_state <= layer_state;
    o_wrfull <= wrfull;
    o_q <= q;
    o_last <= last; 

    gen_tree:
    FOR i in 0 to gen_fifos - 1 GENERATE

        o_mask_n(i) <= i_mask_n(i) or i_mask_n(i + size);
        reset_fifo(i) <= '0' when i_merge_state = '1' or last_layer = '1' else '1';
        
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
                aclr    => not i_reset_n or reset_fifo(i),
                data    => f_data(i),
                rdclk   => i_clk,
                rdreq   => i_rdreq(i),
                wrclk   => i_clk,
                wrreq   => f_wrreq(i),
                q       => last,
                rdempty => o_rdempty(i),
                wrfull  => wrfull(i)--,
            );
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
                aclr    => not i_reset_n or reset_fifo(i),
                data    => f_data(i),
                rdclk   => i_clk,
                rdreq   => i_rdreq(i),
                wrclk   => i_clk,
                wrreq   => f_wrreq(i),
                q       => q(i),
                rdempty => o_rdempty(i),
                wrfull  => wrfull(i)--,
            );
        END GENERATE;
        
        wrreq_good(i) <= '1' when i_merge_state = '1' and wrfull(i) = '0' else '0';
        
        both_rdempty(i) <= '0' when i_rdempty(i) = '0' and i_rdempty(i+size) = '0' else '1';
        
        layer_state(i) <= x"09" when i_merge_state = '0' and last_layer = '1' else

                          x"08" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) = tree_padding and c_h(i) = tree_padding else -- end state
                          x"08" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) = tree_padding and c_z(i) = '1' else -- end state
                          x"08" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_z(i) = '1'          and c_h(i) = tree_padding else -- end state

                          x"00" when wrreq_good(i) = '1' and i_rdempty(i) = '0'      and i_mask_n(i) = '1'      and i_mask_n(i+size) = '0' else
                          x"01" when wrreq_good(i) = '1' and i_rdempty(i+size) = '0' and i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else
                          
                          x"0A" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i) /= tree_padding and c_h(i) = tree_padding and d_h(i) = tree_padding and a_z(i) = '0' and b_z(i) = '0' else
                          x"0B" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and c_h(i) /= tree_padding and d_h(i) /= tree_padding and a_h(i) = tree_padding and b_h(i) = tree_padding and c_z(i) = '0' and d_z(i) = '0' else
                          
                          x"0C" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i) = tree_padding and c_h(i)  = tree_padding and d_h(i) = tree_padding and a_z(i) = '0' else
                          x"0D" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i)  = tree_padding and b_h(i) = tree_padding and c_h(i) /= tree_padding and d_h(i) = tree_padding and c_z(i) = '0' else
                          
                          x"0E" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i) = tree_padding and c_h(i) /= tree_padding and d_h(i) = tree_padding and a(i) <= c(i) and a_z(i) = '0' and c_z(i) = '0' else
                          x"10" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i) = tree_padding and c_h(i) /= tree_padding and d_h(i) = tree_padding and a(i) >  c(i) and a_z(i) = '0' and c_z(i) = '0' else

                          x"11" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i)  = tree_padding and c_h(i) /= tree_padding and d_h(i) /= tree_padding and a(i) <= c(i) and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else                          
                          x"12" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i) /= tree_padding and c_h(i) /= tree_padding and d_h(i)  = tree_padding and c(i) <= a(i) and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          x"13" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i)  = tree_padding and c_h(i) /= tree_padding and d_h(i) /= tree_padding and c(i) <= a(i) and a(i) <= d(i) and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else                          
                          x"14" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a_h(i) /= tree_padding and b_h(i) /= tree_padding and c_h(i) /= tree_padding and d_h(i)  = tree_padding and a(i) <= c(i) and c(i) <= b(i) and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          x"15" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and layer_state_reg(i) /= x"06" and a_h(i) /= tree_padding and b_h(i)  = tree_padding and c_h(i) /= tree_padding and d_h(i) /= tree_padding and c(i) <= a(i) and d(i) <= a(i) and a_z(i) = '0' and c_z(i) = '0' and d_z(i) = '0' else
                          x"16" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and layer_state_reg(i) /= x"07" and a_h(i) /= tree_padding and b_h(i) /= tree_padding and c_h(i) /= tree_padding and d_h(i)  = tree_padding and a(i) <= c(i) and b(i) <= c(i) and a_z(i) = '0' and b_z(i) = '0' and c_z(i) = '0' else
                          
                          x"06" when (layer_state_reg(i) = x"04" or layer_state_reg(i) = x"05") and wrreq_good(i) = '1' and both_rdempty(i) = '0' and b(i) <= d(i) and b_z(i) = '0' and d_z(i) = '0' else
                          x"07" when (layer_state_reg(i) = x"04" or layer_state_reg(i) = x"05") and wrreq_good(i) = '1' and both_rdempty(i) = '0' and b(i) >  d(i) and b_z(i) = '0' and d_z(i) = '0' else
                          
                          x"04" when layer_state_reg(i) = x"06" and wrreq_good(i) = '1' and both_rdempty(i) = '0' and a(i) <= d(i) and a_z(i) = '0' and d_z(i) = '0' else
                          x"0F" when layer_state_reg(i) = x"06" and wrreq_good(i) = '1' and both_rdempty(i) = '0' and a(i) >  d(i) and a_z(i) = '0' and d_z(i) = '0' else
                          
                          x"0F" when layer_state_reg(i) = x"07" and wrreq_good(i) = '1' and both_rdempty(i) = '0' and b(i) <= c(i) and b_z(i) = '0' and c_z(i) = '0' else
                          x"04" when layer_state_reg(i) = x"07" and wrreq_good(i) = '1' and both_rdempty(i) = '0' and b(i) >  c(i) and b_z(i) = '0' and c_z(i) = '0' else
                          
                          x"04" when layer_state_reg(i) = x"04" else
                          x"05" when layer_state_reg(i) = x"05" else
                          x"06" when layer_state_reg(i) = x"06" else
                          x"07" when layer_state_reg(i) = x"07" else
                          
                          x"02" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a(i) <= c(i) and b(i) <= c(i) and a_z(i) = '0' and b_z(i) = '0' else
                          x"03" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and c(i) <= a(i) and d(i) <= a(i) and c_z(i) = '0' and d_z(i) = '0' else  
                          x"04" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and a(i) <= c(i) and b(i) >  c(i) and a_z(i) = '0' and c_z(i) = '0' else
                          x"05" when wrreq_good(i) = '1' and both_rdempty(i) = '0' and c(i) <= a(i) and d(i) >  a(i) and c_z(i) = '0' and a_z(i) = '0' else
                          
                          x"0F";
                         
        wrreq(i) <= '1' when layer_state(i) = x"09" and i_wen_h_t = '1' else
                    '1' when layer_state(i) = x"00" or layer_state(i) = x"01" or layer_state(i) = x"02" or layer_state(i) = x"03" or layer_state(i) = x"04" or layer_state(i) = x"05" or layer_state(i) = x"08" or layer_state(i) = x"0A" or layer_state(i) = x"0B" or layer_state(i) = x"0C" or layer_state(i) = x"0D" or layer_state(i) = x"0E" or layer_state(i) = x"10" or layer_state(i) = x"13" or layer_state(i) = x"14" or layer_state(i) = x"15" or layer_state(i) = x"16" else
                    '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else
                    '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else
                    '1' when layer_state(i) = x"04" and layer_state_reg(i) = x"06" else
                    '1' when layer_state(i) = x"04" and layer_state_reg(i) = x"07" else
                    '1' when layer_state(i) = x"0C" and layer_state_reg(i) = x"11" else
                    '1' when layer_state(i) = x"0C" and layer_state_reg(i) = x"12" else
                    '1' when layer_state(i) = x"0D" and layer_state_reg(i) = x"11" else
                    '1' when layer_state(i) = x"0D" and layer_state_reg(i) = x"11" else
                    '0';
                    
        o_rdreq(i) <= '1' when layer_state(i) = x"00" or layer_state(i) = x"02" or layer_state(i) = x"06" or layer_state(i) = x"0A" or layer_state(i) = x"0C" or layer_state(i) = x"0E" or layer_state(i) = x"10" or layer_state(i) = x"11" or layer_state(i) = x"13" or layer_state(i) = x"16" else
                      '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else
                      '1' when layer_state(i) = x"06" and layer_state_reg(i) = x"04" else
                      '1' when layer_state(i) = x"06" and layer_state_reg(i) = x"05" else
                      '0';

        o_rdreq(i+size) <=  '1' when layer_state(i) = x"01" or layer_state(i) = x"03" or layer_state(i) = x"07" or layer_state(i) = x"0B" or layer_state(i) = x"0D" or layer_state(i) = x"0E" or layer_state(i) = x"10" or layer_state(i) = x"12" or layer_state(i) = x"14" or layer_state(i) = x"15" else
                            '1' when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else
                            '1' when layer_state(i) = x"07" and layer_state_reg(i) = x"04" else
                            '1' when layer_state(i) = x"07" and layer_state_reg(i) = x"05" else
                            '0';
        
        data(i)(37 downto 0) <= i_data_h_t when layer_state(i) = x"09" else
                                tree_padding when layer_state(i) = x"08" else
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
                                data_reg(i)(37 downto 0) when layer_state(i) = x"0C" and layer_state_reg(i) = x"11" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"0C" and layer_state_reg(i) = x"12" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"0D" and layer_state_reg(i) = x"11" else
                                data_reg(i)(37 downto 0) when layer_state(i) = x"0D" and layer_state_reg(i) = x"12" else
                                a_h(i) when layer_state(i) = x"00" or layer_state(i) = x"02" or layer_state(i) = x"04" or layer_state(i) = x"0A" or layer_state(i) = x"0C" or layer_state(i) = x"0E" or layer_state(i) = x"11" or layer_state(i) = x"14" or layer_state(i) = x"16" else
                                c_h(i) when layer_state(i) = x"01" or layer_state(i) = x"03" or layer_state(i) = x"05" or layer_state(i) = x"0B" or layer_state(i) = x"0D" or layer_state(i) = x"10" or layer_state(i) = x"12" or layer_state(i) = x"13" or layer_state(i) = x"15" else
                                b_h(i) when layer_state(i) = x"06" else
                                d_h(i) when layer_state(i) = x"07" else
                                (others => '0');

        data(i)(75 downto 38)<= tree_paddingk when layer_state(i) = x"09" else
                                tree_padding when layer_state(i) = x"08" or layer_state(i) = x"0C" or layer_state(i) = x"0D" else
                                data_reg(i)(75 downto 38) when layer_state(i) = x"06" and layer_state_reg(i) = x"06" else
                                data_reg(i)(75 downto 38) when layer_state(i) = x"07" and layer_state_reg(i) = x"07" else
                                a_h(i) when layer_state(i) = x"0C" and layer_state_reg(i) = x"11" else
                                a_h(i) when layer_state(i) = x"0C" and layer_state_reg(i) = x"12" else
                                c_h(i) when layer_state(i) = x"0D" and layer_state_reg(i) = x"11" else
                                c_h(i) when layer_state(i) = x"0D" and layer_state_reg(i) = x"12" else
                                d_h(i) when layer_state(i) = x"0F" and layer_state_reg(i) = x"06" else 
                                b_h(i) when layer_state(i) = x"0F" and layer_state_reg(i) = x"07" else 
                                a_h(i) when layer_state(i) = x"04" and layer_state_reg(i) = x"06" else
                                c_h(i) when layer_state(i) = x"04" and layer_state_reg(i) = x"07" else
                                b_h(i) when layer_state(i) = x"00" or layer_state(i) = x"02" or layer_state(i) = x"0A" or layer_state(i) = x"16" else
                                d_h(i) when layer_state(i) = x"15" or layer_state(i) = x"01" or layer_state(i) = x"03" or layer_state(i) = x"0B" else
                                a_h(i) when layer_state(i) = x"05" or layer_state(i) = x"10" or layer_state(i) = x"13" else
                                c_h(i) when layer_state(i) = x"04" or layer_state(i) = x"0E" or layer_state(i) = x"14" else
                                (others => '0');
        
        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n /= '1' ) then
            layer_state_reg(i) <= (others => '0');
        elsif ( rising_edge(i_clk) ) then
            layer_state_reg(i) <= layer_state(i);
            data_reg(i) <= data(i);
        end if;
        end process;
        
        -- reg for FIFO inputs (timing)
        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n /= '1' ) then
            f_data(i)   <= (others => '0');
            f_wrreq(i)  <= '0';
        elsif ( rising_edge(i_clk) ) then
            f_data(i)   <= data(i);
            f_wrreq(i)  <= wrreq(i);
        end if;
        end process;

    END GENERATE;

end architecture;
