library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.dataflow_components.all;

-- merge packets delimited by SOP and EOP from N input streams
entity time_merger is
generic (
    W : positive := 32;
    TIMEOUT : std_logic_vector(15 downto 0) := x"FFFF";
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    data_array(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0); -- start of packet (SOP)
    i_reop      : in    std_logic_vector(N-1 downto 0); -- end of packet (EOP)
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    i_mask_n    : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0); -- read ACK

    -- output stream
    o_wdata     : out   std_logic_vector(W-1 downto 0);
    o_wsop      : out   std_logic; -- SOP
    o_weop      : out   std_logic; -- EOP
    i_wfull     : in    std_logic;
    o_we        : out   std_logic; -- write enable

    -- error outputs
    o_error_pre : out std_logic_vector(N - 1 downto 0);
    o_error_sh : out std_logic_vector(N - 1 downto 0);
    o_error_gtime : out std_logic;
    o_error_shtime : out std_logic;
    
    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of time_merger is

    type data_array_t is array (natural range <>) of std_logic_vector(W - 1 downto 0);
    type fpga_id_array_t is array (natural range <>) of std_logic_vector(15 downto 0);
    type sheader_time_array_t is array (natural range <>) of std_logic_vector(9 downto 0);
    signal rdata : data_array_t(N - 1 downto 0);

    -- merge state
    signal error_gtime, error_shtime : std_logic;
    type merge_state_type is (wait_for_pre, compare_time1, compare_time2, wait_for_sh, error_state, merge_hits);
    signal merge_state : merge_state_type;
    signal check_pre, check_sh, check_time1, check_time2, error_pre, error_sh : std_logic_vector(N - 1 downto 0);
    signal gtime1, gtime2 : std_logic_vector(31 downto 0);
    constant check_zeros : std_logic_vector(check_pre'range) := (others => '0');
    signal sheader_time : sheader_time_array_t(N - 1 downto 0);
    signal fpga_id, wait_cnt_pre, wait_cnt_sh : fpga_id_array_t(N - 1 downto 0);

begin

    generate_rdata : for i in 0 to N-1 generate
        rdata(i) <= i_rdata(i);
    end generate;
    
    o_error_gtime <= error_gtime;
    o_error_shtime <= error_shtime;
    o_error_pre <= error_pre;
    o_error_sh <= error_sh;

    process(i_clk, i_reset_n)
        variable cur_time : std_logic_vector(31 downto 0);
    begin
    if ( i_reset_n /= '1' ) then
        merge_state <= wait_for_pre;
        
        check_pre <= (others => '1');
        check_sh <= (others => '1');
        check_time1 <= (others => '1');
        check_time2 <= (others => '1');
        error_pre <= (others => '0');
        error_sh <= (others => '0');
        wait_cnt_pre <= (others => (others => '0'));
        wait_cnt_sh <= (others => (others => '0'));
        fpga_id <= (others => (others => '0'));
        gtime1 <= (others => '0');
        gtime2 <= (others => '0');
        sheader_time <= (others => (others => '0'));
        
        o_wdata <= (others => '0');
        o_rack <= (others => '0');
        o_wsop <= '0';
        o_weop <= '0';
        o_we <= '0';
        --
    elsif rising_edge(i_clk) then
        
        o_rack <= (others => '0');
        o_we <= '0';
        o_wsop <= '0';
        o_weop <= '0';
        o_wdata <= (others => '0');
    
        case merge_state is
            -- readout until all fifos have preamble
            when wait_for_pre =>
                
                FOR I in N - 1 downto 0 LOOP
                    if ( wait_cnt_pre(I) = TIMEOUT ) then
                        error_pre(I) <= '1';
                    end if;
                    
                    if ( i_rempty(I) = '0' and i_rdata(I)(35 downto 30) = "111010" ) then
                        check_pre(I) <= '0';
                        fpga_id(I) <= i_rdata(I)(27 downto 12);
                        o_rack(I) <= '1';
                    elsif ( i_mask_n(I) = '0' ) then
                        check_pre(I) <= '0';
                    else
                        wait_cnt_pre(I) <= wait_cnt_pre(I) + '1';
                        o_rack(I) <= '1';
                    end if;
                END LOOP;
                
                -- check if fifo is not full and all links have preamble
                if( check_pre = check_zeros and i_wfull = '0' ) then
                    merge_state <= compare_time1;
                    -- reset signals
                    wait_cnt_pre <= (others => (others => '0'));
                    check_pre <= (others => '1');
                    -- send merged data preamble
                    -- sop & preamble & zeros & datak
                    o_wdata(37 downto 36) <= "01";
                    o_wdata(35 downto 30) <= "111010";
                    o_wdata(11 downto 4) <= x"BC";
                    o_wdata(3 downto 0) <= "0001";
                    o_we <= '1';
                end if;
                
            when compare_time1 =>
                if ( error_pre /= check_zeros ) then
                    merge_state <= error_state;
                end if;
                FOR I in N - 1 downto 0 LOOP
                    FOR j in N - 1 downto 0 LOOP
                        if ( i_rempty(I) = '0' and i_mask_n(I) = '1' ) then
                            o_rack(I) <= '1';
                            if ( i_rempty(J) = '0' and i_mask_n(J) = '1' ) then
                                if ( i_rdata(I) /= i_rdata(J) ) then
                                    error_gtime <= '1';
                                else
                                    -- send gtime
                                    check_time1(I) <= '0';
                                    gtime1 <= i_rdata(I)(35 downto 4);
                                end if;
                            end if;
                        end if;
                    END LOOP;
                END LOOP;
                
                 -- check if fifo is not full and all links have same time
                if ( check_time1 = check_zeros and i_wfull = '0' ) then
                    merge_state <= compare_time2;
                    -- reset signals
                    check_time1 <= (others => '1');
                    -- send gtime1
                    o_wdata(37 downto 36) <= "00";
                    o_wdata(35 downto 4) <= gtime1;
                    o_wdata(3 downto 0) <= "0000";
                    o_we <= '1';
                end if;
                
            when compare_time2 =>
                if ( error_gtime = '1' ) then
                    merge_state <= error_state;
                end if;
                FOR I in N - 1 downto 0 LOOP
                    FOR j in N - 1 downto 0 LOOP
                        if ( i_rempty(I) = '0' and i_mask_n(I) = '1' ) then
                            o_rack(I) <= '1';
                            if ( i_rempty(J) = '0' and i_mask_n(J) = '1' ) then
                                if ( i_rdata(I) /= i_rdata(J) ) then
                                    error_gtime <= '1';
                                else
                                    -- send gtime
                                    check_time2(I) <= '0';
                                    gtime2 <= i_rdata(I)(35 downto 4);
                                end if;
                            end if;
                        end if;
                    END LOOP;
                END LOOP;
                
                -- check if fifo is not full and all links have same time
                if ( check_time2 = check_zeros and i_wfull = '0' ) then
                    merge_state <= wait_for_sh;
                    -- reset signals
                    check_time2 <= (others => '1');
                    -- send gtime2
                    o_wdata(37 downto 36) <= "00";
                    o_wdata(35 downto 4) <= gtime2;
                    o_wdata(3 downto 0) <= "0000";
                    o_we <= '1';
                end if;
                
            when wait_for_sh =>
                if ( error_gtime = '1' ) then
                    merge_state <= error_state;
                else
                    -- check for sub header
                    FOR I in N - 1 downto 0 LOOP
                        if ( wait_cnt_sh(I) = TIMEOUT ) then
                            error_pre(I) <= '1';
                        end if;
                        
                        if ( i_rempty(I) = '0' and i_rdata(I)(31 downto 26) = "111111" ) then
                            check_sh(I) <= '0';
                            sheader_time(I) <= i_rdata(I)(25 downto 16);
                            o_rack(I) <= '1';
                        elsif ( i_mask_n(I) = '0' ) then
                            check_sh(I) <= '0';
                        else
                            wait_cnt_sh(I) <= wait_cnt_sh(I) + '1';
                            o_rack(I) <= '1';
                        end if;
                    END LOOP;
                end if;
                
                -- check if fifo is not full and all links have subheader
                if( check_sh = check_zeros and i_wfull = '0' ) then
                    merge_state <= merge_hits;
                    -- reset signals
                    wait_cnt_sh <= (others => (others => '0'));
                    check_sh <= (others => '1');
                    -- send merged data sub header
                    -- zeros & sub header & zeros & datak
                    o_wdata(37 downto 36) <= "00";
                    o_wdata(35 downto 32) <= "0000";
                    o_wdata(31 downto 26) <= "111111";
                    o_wdata(3 downto 0) <= "0001";
                    o_we <= '1';
                    -- check if sheader time is equal
                    FOR I in N - 1 downto 0 LOOP
                        FOR j in N - 1 downto 0 LOOP
                            if ( i_rempty(I) = '0' and i_mask_n(I) = '1' ) then
                                if ( i_rempty(J) = '0' and i_mask_n(J) = '1' ) then
                                    if ( sheader_time(I) /= sheader_time(J) ) then
                                        error_shtime <= '1';
                                    else
                                        -- send sub header time
                                        o_wdata(25 downto 16) <= sheader_time(I);
                                    end if;
                                end if;
                            end if;
                        END LOOP;
                    END LOOP;
                end if;
                
            when merge_hits =>
                if ( error_shtime = '1' ) then
                    merge_state <= error_state;
                else
                    --
                end if;
                
            when error_state =>
                -- send error message xxxxxxDC
                -- 12: error gtime
                -- 13: error shtime
                -- N+14 downto 14: error wait for pre
                o_wdata(3 downto 0) <= "0001";
                o_wdata(11 downto 4) <= x"DC";
                o_wdata(12) <= error_gtime;
                o_wdata(12) <= error_shtime;
                o_wdata(N + 14 downto 14) <= error_pre;
                o_we <= '1';
                o_weop <= '1';
                
            when others =>
                merge_state <= wait_for_pre;
                
        end case;
        --
    end if;
    end process;

end architecture;
