library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


entity time_merger_tree_fifo_32_v3 is
generic (
    g_NLINKS_DATA   : positive  := 12;
    TREE_w          : positive  := 8;
    TREE_r          : positive  := 8;
    compare_fifos   : positive  := 16;
    gen_fifos       : positive  := 16;
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE : std_logic_vector(7 downto 0) := x"01"--;
);
port (
    -- input data stream
    i_data          : in  work.util.slv32_array_t(compare_fifos - 1 downto 0);
    i_rshop         : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_rsop          : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_reop          : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_hit           : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_t0            : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_t1            : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_rempty        : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_mask_n        : in  std_logic_vector(compare_fifos - 1 downto 0);
    i_en            : in  std_logic;
    o_rack          : out std_logic_vector(compare_fifos - 1 downto 0);

    -- output data stream
    o_data          : out work.util.slv32_array_t(gen_fifos - 1 downto 0);
    o_rshop         : out std_logic_vector(gen_fifos - 1 downto 0);
    o_rsop          : out std_logic_vector(gen_fifos - 1 downto 0);
    o_reop          : out std_logic_vector(gen_fifos - 1 downto 0);
    o_hit           : out std_logic_vector(gen_fifos - 1 downto 0);
    o_t0            : out std_logic_vector(gen_fifos - 1 downto 0);
    o_t1            : out std_logic_vector(gen_fifos - 1 downto 0);
    o_rempty        : out std_logic_vector(gen_fifos - 1 downto 0);
    o_mask_n        : out std_logic_vector(gen_fifos - 1 downto 0);
    i_rack          : in  std_logic_vector(gen_fifos - 1 downto 0);

    -- error out
    o_error         : out work.util.slv32_array_t(compare_fifos - 1 downto 0);

    i_reset_n       : in  std_logic;
    i_clk           : in  std_logic--;
);
end entity;

architecture arch of time_merger_tree_fifo_32_v3 is

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

    signal data, q_data : work.util.slv35_array_t(gen_fifos - 1 downto 0);
    signal layer_state : work.util.slv8_array_t(gen_fifos - 1 downto 0);
    signal wrreq, wrfull, reset_fifo, wrfull_and_merge_state, both_inputs_rdempty, rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal wrfull_and_merge_state_and_both_inputs_not_rdempty, wrfull_and_merge_state_and_first_input_not_rdempty, wrfull_and_merge_state_and_second_input_not_rdempty : std_logic_vector(gen_fifos - 1 downto 0);
    signal first_input_mask_n_second_input_not_mask_n, second_input_mask_n_first_input_not_mask_n, a_padding, b_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a_b_padding, b_no_a_padding, a_no_b_padding : std_logic_vector(gen_fifos - 1 downto 0);
    signal a, b : work.util.slv4_array_t(gen_fifos - 1 downto 0) := (others => (others => '1'));
    signal a_h, b_h : work.util.slv32_array_t(gen_fifos - 1 downto 0) := (others => (others => '1'));
    signal last : std_logic_vector(r_width-1 downto 0);
    signal last_data : std_logic_vector(8 * 32 - 1 downto 0);
    signal last_link : std_logic_vector(8 *  6 - 1 downto 0);
    signal wrfull_last0, wrfull_last1, wrfull_last2, wrfull_s, rdempty_last0, rdempty_last1, rdempty_last2 : std_logic_vector(gen_fifos - 1 downto 0);

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

        a_h(i)      <= i_data(i);
        b_h(i)      <= i_data(i+size);

        o_mask_n(i) <= i_mask_n(i) or i_mask_n(i + size);

        e_last_fifo_link_debug : entity work.ip_scfifo_v2
        generic map (
            g_ADDR_WIDTH => 11,
            g_DATA_WIDTH => 35,
            g_WREG_N => 1, -- TNS=...
            g_RREG_N => 1--, -- TNS=-2300
        )
        port map (
            i_wdata         => data(i),
            i_we            => wrreq(i),
            o_wfull         => wrfull(i),

            o_rdata         => q_data(i),
            o_rempty        => o_rempty(i),
            i_rack          => i_rack(i),

            i_clk           => i_clk,
            i_reset_n       => i_reset_n--,
        );

        o_rsop(i)   <= '1' when q_data(i)(40 downto 38) = "010" else '0';
        o_rshop(i)  <= '1' when q_data(i)(40 downto 38) = "111" else '0';
        o_reop(i)   <= '1' when q_data(i)(40 downto 38) = "001" else '0';
        o_rhit(i)   <= '1' when q_data(i)(40 downto 38) = "000" else '0';
        o_rt0(i)    <= '1' when q_data(i)(40 downto 38) = "100" else '0';
        o_rt1(i)    <= '1' when q_data(i)(40 downto 38) = "101" else '0';
        o_data(i)   <= q_data(i)(37 downto 0);

        -- combine mask and empty 
        empty_or_mask(i)      <= i_rdempty(i)      or not i_rmask_n(i)      when i_en = '1' else '1';
        empty_or_mask(i+size) <= i_rdempty(i+size) or not i_rmask_n(i+size) when i_en = '1' else '1';

        -- Tree setup
        -- x => empty, h => header, t => time header, tr => trailer, sh => sub header
        -- [a]               [a]                   [a]
        -- [1]  -> [[2],[1]] [tr]  -> [[tr],[2]]   [4,sh]   -> [[4],[3],[sh],[2]]
        -- [2]               [tr,2]                [3,sh,2]
        -- [b]               [b]                   [b]

        --! write only if not idle
        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n /= '1' ) then
            data(i)         <= (others => '0');
            cnt_events(i)   <= (others => '0');
            wrreq(i)        <= '0';
            o_rack(i)       <= '0';
            o_rack(i+size)  <= '0';
            layer_state(i)  <= IDEL;
            --
        elsif ( rising_edge(i_clk) ) then

            data(i)         <= (others => '0');
            wrreq(i)        <= '0';
            o_rack(i)       <= '0';
            o_rack(i+size)  <= '0';

            if ( (i_rmask_n(i) = '1' and i_rmask_n(i+size) = '1') or
                 i_rdempty(i) = '1' or i_rdempty(i+size) = '1' or wrfull(i) = '1' or i_en = '0'
            ) then
                --
            else
                case layer_state(i) is

                when IDEL =>
                    if ( (i_sop(i)      = '1' and i_rmask_n(i)      = '0') or 
                         (i_sop(i+size) = '1' and i_rmask_n(i+size) = '0') or 
                         (i_sop(i)      = '1' and i_sop(i+size) = '1')
                    ) then
                        o_rack(i)           <= not i_rmask_n(i);
                        o_rack(i+size)      <= not i_rmask_n(i+size);
                        data(35 downto 32)  <= i_sop(i);
                        data(31 downto 26)  <= "111010";
                        data(7 downto 0)    <= x"BC";
                        wrreq(i)            <= '1';
                        cnt_events(i)       <= cnt_events(i) + '1';
                        layer_state(i)      <= write_ts_0;
                    end if;

                when write_ts_0 =>
                    link_to_fifo_state          <= write_ts_1;
                    rx_156_data(34 downto 32)   <= "100"; -- ts0
                    rx_156_wen <= '1';

                when write_ts_1 =>
                    link_to_fifo_state <= write_data;
                    rx_156_data(34 downto 32)   <= "101"; -- ts1
                    rx_156_wen <= '1';

                when write_data =>
                    if ( i_rx(7 downto 0) = x"9C" and i_rx_k = "0001" ) then
                        link_to_fifo_state <= idle;
                        rx_156_data(34 downto 32) <= "001"; -- trailer
                    end if;

                    if ( i_rx(31 downto 26) = "111111" and i_rx_k = "0000" ) then
                        rx_156_data(34 downto 32) <= "111"; -- sub header
                        cnt_sub <= cnt_sub + '1';
                    end if;

                    hit_reg <= i_rx;

                    if ( SKIP_DOUBLE_SUB = 1 and i_rx = hit_reg ) then
                        rx_156_wen <= '0';
                    else
                        rx_156_wen <= '1';
                    end if;

                when skip_data =>
                    if ( i_rx(7 downto 0) = x"9C" and i_rx_k = "0001" ) then
                        link_to_fifo_state <= idle;
                    end if;

                when others =>
                    link_to_fifo_state <= idle;

                end case;
            end if;
            --
        end if;
        end process;
        

        -- construct layer states
        first_input_mask_n_second_input_not_mask_n(i)           <= '1' when i_mask_n(i) = '1'      and i_mask_n(i+size) = '0' else '0';
        second_input_mask_n_first_input_not_mask_n(i)           <= '1' when i_mask_n(i+size) = '1' and i_mask_n(i)      = '0' else '0';

        both_inputs_rdempty(i)                                  <= '1' when i_rdempty(i) = '1' and i_rdempty(i+size) = '1' else '0';
        both_inputs_not_rdempty(i)                              <= '1' when i_rdempty(i) = '0' and i_rdempty(i+size) = '0' else '1';
        first_input_not_rdempty(i)                              <= '1' when i_rdempty(i) = '0' and i_rdempty(i+size) = '1' else '0';
        second_input_not_rdempty(i)                             <= '1' when i_rdempty(i) = '1' and i_rdempty(i+size) = '0' else '0';

        both_hit(i)                                             <= '1' when i_hit(i)  = '1' and i_hit(i+size)  = '1' else '0';
        first_hit(i)
        second_hit(i)
        
        both_sop(i)                                             <= '1' when i_sop(i)  = '1' and i_sop(i+size)  = '1' else '0';
        first_sop(i)                                            <= '1' when i_sop(i)  = '1' and i_sop(i+size)  = '1' else '0';
        second_sop(i)                                           <= '1' when i_sop(i)  = '1' and i_sop(i+size)  = '1' else '0';
        
        both_eop(i)                                             <= '1' when i_eop(i)  = '1' and i_eop(i+size)  = '1' else '0';
        
        
        both_shop(i)                                            <= '1' when i_shop(i) = '1' and i_shop(i+size) = '1' else '0';
        
        
        both_t0(i)                                              <= '1' when i_t0(i)   = '1' and i_t0(i+size)   = '1' else '0';
        
        
        both_t1(i)                                              <= '1' when i_t1(i)   = '1' and i_t1(i+size)   = '1' else '0';

        
        second_hit(i)
        

        -- TODO: include sub-header, check backpres., counters etc.
        layer_state(i) <=   fifo_full_state when wrfull(i) = '1' else
                            both_empty when both_inputs_rdempty(i) = '1' else
                            
                            second_input_not_mask_n when  first_input_mask_n_second_input_not_mask_n(i) = '1' else
                            first_input_not_mask_n  when  second_input_mask_n_first_input_not_mask_n(i) = '1' else
                            
                            
                            a_smaller_b when both_inputs_not_rdempty(i) = '1' and both_hit(i) = '1' and a(i) <= b(i) else
                            b_smaller_a when both_inputs_not_rdempty(i) = '1' and both_hit(i) = '1' and a(i)  > b(i) else

                            IDEL;

        wrreq(i)        <=  '0' layer_state(i) = fifo_full_state(i)      else
                            '0' layer_state(i) = both_empty              else
                            '1' layer_state(i) = first_input_not_mask_n  else
                            '1' layer_state(i) = second_input_not_mask_n else
                            
        
                            '1' when layer_state(i) = end_state and wrfull_s(i) = '0' else
                            '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = first_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = b_no_a_padding_state or layer_state(i) = a_smaller_b or layer_state(i) = b_smaller_a else
                            '0';

        o_rdreq(i)      <=  '0' layer_state(i) = fifo_full_state(i)         else
                            '0' layer_state(i) = both_empty                  else
                            '1' layer_state(i) = first_input_not_mask_n(i)  else
        
        
        o_rdreq(i+size) <=  '0' layer_state(i) = fifo_full_state(i)         else
                            '0' both_inputs_rdempty(i) = '1'                else
                            '1' layer_state(i) = second_input_not_mask_n(i) else
        
        
         o_rdreq(i)      <= '1' when layer_state(i) = second_input_not_mask_n or layer_state(i) = a_no_b_padding_state or layer_state(i) = a_smaller_b else
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
