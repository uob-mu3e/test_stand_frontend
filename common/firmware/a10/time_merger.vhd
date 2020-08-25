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
    MERGER_SPLIT : positive := 4;
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
    o_error_gtime : out std_logic_vector(1 downto 0);
    o_error_shtime : out std_logic;
    
    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of time_merger is

    -- find min index
    function get_min_index (
        msb : integer;
        lsb : integer;
        good : std_logic_vector;
        minval : std_logic_vector;
        data : data_array--;
    ) return integer is
        variable index : integer := 999;
        variable min_value : std_logic_vector(4 downto 0);
    begin
        min_value := minval;
        FOR I in msb - 1 downto lsb LOOP
            if ( data(I)(37 downto 36) = "00" and 
                 data(I)(31 downto 26) /= "111111" and
                 good(I) = '0' and
                 data(I)(35 downto 32) < min_value
            ) then
                min_value(3 downto 0) := data(I)(35 downto 32);
                min_value(4) := '0';
                index := I;
            end if;
        END LOOP;
        return index;
    end function;
    
    -- get hit time
    function get_hit_time (
        index : integer;
        data : data_array--;
    ) return std_logic_vector is
        variable min_value : std_logic_vector(4 downto 0);
    begin
        if ( index = 999 ) then
            min_value := "11111";
        else
            min_value(3 downto 0) := data(index)(35 downto 32);
            min_value(4) := '0';
        end if;
        return min_value;
    end function;
 
    type fpga_id_array_t is array (N - 1 downto 0) of std_logic_vector(15 downto 0);
    type sheader_time_array_t is array (N - 1 downto 0) of std_logic_vector(5 downto 0);
    type ram_add_t is array (N - 1 downto 0) of std_logic_vector(7 downto 0);
    type merge_state_type is (wait_for_pre, compare_time1, compare_time2, wait_for_sh, error_state, merge_hits, get_time1, get_time2, trailer);
    
    constant check_zeros : std_logic_vector(N - 1 downto 0) := (others => '0');
    constant check_ones : std_logic_vector(N - 1 downto 0) := (others => '1');
    constant all_zeros : ram_add_t := (others => (others => '0'));

    signal error_gtime1, error_gtime2, error_shtime, w_ack : std_logic;
    signal merge_state : merge_state_type;
    signal rack, check_pre, check_sh, check_tr, check_time1, check_time2, error_pre, error_sh, error_tr, cnt_sh_header, cnt_pre_header, cnt_trailer, cnt_ram_start : std_logic_vector(N - 1 downto 0);
    signal gtime1, gtime2 : std_logic_vector(31 downto 0);
    signal shtime : std_logic_vector(5 downto 0);
    signal sheader_time : sheader_time_array_t;
    signal fpga_id, wait_cnt_pre, wait_cnt_sh, wait_cnt_tr : fpga_id_array_t;
    
    -- merge signals
    signal min_hit : std_logic_vector(37 downto 0);
    signal link_good : std_logic_vector(N - 1 downto 0);

begin

    o_rack <= rack;
    o_error_gtime(0) <= error_gtime1;
    o_error_gtime(1) <= error_gtime2;
    o_error_shtime <= error_shtime;
    o_error_pre <= error_pre;
    o_error_sh <= error_sh;
    
    -- good = 0 if link is not empty, wfull not full, not masked and w_ack is zero
    generate_link_good : FOR I in N-1 downto 0 GENERATE
        link_good(I) <= i_rempty(I) or i_wfull or not i_mask_n(I) or w_ack;
    END GENERATE;
    
    process(i_clk, i_reset_n)
        -- min value has one more bit than hit time
        variable min_value1 : std_logic_vector(4 downto 0);
        variable min_index1 : integer;
        variable min_value2 : std_logic_vector(4 downto 0);
        variable min_index2 : integer;
        variable min_value3 : std_logic_vector(4 downto 0);
        variable min_index3 : integer;
        variable min_value4 : std_logic_vector(4 downto 0);
        variable min_index4 : integer;
    begin
    if ( i_reset_n /= '1' ) then
        merge_state <= wait_for_pre;
        check_pre <= (others => '1');
        check_sh <= (others => '1');
        check_tr <= (others => '1');
        check_time1 <= (others => '1');
        check_time2 <= (others => '1');
        error_pre <= (others => '0');
        error_sh <= (others => '0');
        error_tr <= (others => '0');
        wait_cnt_pre <= (others => (others => '0'));
        wait_cnt_sh <= (others => (others => '0'));
        wait_cnt_tr <= (others => (others => '0'));
        fpga_id <= (others => (others => '0'));
        gtime1 <= (others => '1');
        gtime2 <= (others => '1');
        shtime <= (others => '1');
        sheader_time <= (others => (others => '0'));
        error_gtime1 <= '0';
        error_gtime2 <= '0';
        error_shtime <= '0';
        w_ack <= '0';
        min_value1 := (others => '0');
        min_value2 := (others => '0');
        min_value3 := (others => '0');
        min_value4 := (others => '0');
        min_index1 := 0;
        min_index2 := 0;
        min_index3 := 0;
        min_index4 := 0;
        min_hit <= (others => '0');
        cnt_sh_header <= (others => '1');
        cnt_pre_header <= (others => '1');
        cnt_trailer <= (others => '1');
        cnt_ram_start <= (others => '1');
        o_wdata <= (others => '0');
        rack <= (others => '0');
        o_wsop <= '0';
        o_weop <= '0';
        o_we <= '0';
        --
    elsif rising_edge(i_clk) then
        
        rack <= (others => '0');
        o_we <= '0';
        o_wsop <= '0';
        o_weop <= '0';
    
        case merge_state is
            -- readout until all fifos have preamble
            when wait_for_pre =>
                
                FOR I in N - 1 downto 0 LOOP
                    if ( i_mask_n(I) = '0' ) then
                        check_pre(I) <= '0';
                    end if;
                    
                    if ( wait_cnt_pre(I) = TIMEOUT ) then
                        error_pre(I) <= '1';
                    end if;
                    
                    if ( i_rdata(I)(35 downto 30) = "111010" and i_rdata(I)(37 downto 36) = "01" and check_pre(I) = '1' ) then
                        check_pre(I) <= i_rempty(I);
                        fpga_id(I) <= i_rdata(I)(27 downto 12);
                        rack(I) <= not i_rempty(I);
                    end if;
                    
                    if ( check_pre(I) = '1' ) then
                        wait_cnt_pre(I) <= wait_cnt_pre(I) + '1';
                        rack(I) <= not i_rempty(I);
                    end if;
                END LOOP;
                
                -- check if fifo is not full and all links have preamble
                if( check_pre = check_zeros and i_wfull = '0' ) then
                    merge_state <= get_time1;
                    -- reset signals
                    wait_cnt_pre <= (others => (others => '0'));
                    check_pre <= (others => '1');
                    -- send merged data preamble
                    -- sop & preamble & zeros & datak
                    o_wdata(37 downto 36) <= "01";
                    o_wdata(35 downto 30) <= "111010";
                    o_wdata(11 downto 4) <= x"BC";
                    o_wdata(3 downto 0) <= "0001";
                    o_wsop <= '1';
                    o_we <= '1';
                end if;
                
            when get_time1 =>
                FOR I in N - 1 downto 0 LOOP
                    if ( i_rempty(I) = '0' and i_mask_n(I) = '1' ) then
                        merge_state <= compare_time1;
                        gtime1 <= i_rdata(I)(35 downto 4);
                        exit;
                    end if;
                END LOOP;
                
            when compare_time1 =>
                if ( error_pre /= check_zeros ) then
                    merge_state <= error_state;
                end if;
                
                FOR I in N - 1 downto 0 LOOP
                    if ( i_mask_n(I) = '0' ) then
                        check_time1(I) <= '0';
                    end if;
                    
                    if ( i_rdata(I)(35 downto 4) /= gtime1 and i_mask_n(I) = '1' ) then
                        error_gtime1 <= '1';
                    elsif ( check_time1(I) = '1' ) then
                        -- check gtime
                        check_time1(I) <= i_rempty(I);
                        rack(I) <= not i_rempty(I);
                    end if;
                END LOOP;
                 
                 -- check if fifo is not full and all links have same time
                if ( check_time1 = check_zeros and i_wfull = '0' ) then
                    merge_state <= get_time2;
                    -- reset signals
                    check_time1 <= (others => '1');
                    gtime1 <= (others => '0');
                    -- send gtime1
                    o_wdata(37 downto 36) <= "00";
                    o_wdata(35 downto 4) <= gtime1;
                    o_wdata(3 downto 0) <= "0000";
                    o_we <= '1';
                end if;
                
            when get_time2 =>
                FOR I in N - 1 downto 0 LOOP
                    if ( i_rempty(I) = '0' and i_mask_n(I) = '1' ) then
                        merge_state <= compare_time2;
                        gtime2 <= i_rdata(I)(35 downto 4);
                        exit;
                    end if;
                END LOOP;
                
            when compare_time2 =>
                if ( error_gtime1 = '1' ) then
                    merge_state <= error_state;
                end if;

                FOR I in N - 1 downto 0 LOOP
                    if ( i_mask_n(I) = '0' ) then
                        check_time1(I) <= '0';
                    end if;
                    
                    if ( i_rdata(I)(35 downto 4) /= gtime2 and i_mask_n(I) = '1' ) then
                        error_gtime2 <= '1';
                    elsif ( check_time2(I) = '1' ) then
                        -- send gtime
                        check_time2(I) <= i_rempty(I);
                        rack(I) <= not i_rempty(I);
                    end if;
                END LOOP;

                -- check if fifo is not full and all links have same time
                if ( check_time2 = check_zeros and i_wfull = '0' ) then
                    merge_state <= wait_for_sh;
                    -- reset signals
                    check_time2 <= (others => '1');
                    gtime2 <= (others => '0');
                    -- send gtime2
                    o_wdata(37 downto 36) <= "00";
                    o_wdata(35 downto 4) <= gtime2;
                    o_wdata(3 downto 0) <= "0000";
                    o_we <= '1';
                end if;
                
            when wait_for_sh =>
                if ( error_gtime2 = '1' ) then
                    merge_state <= error_state;
                end if;
                
                -- check for sub header
                FOR I in N - 1 downto 0 LOOP
                    if ( i_mask_n(I) = '0' ) then
                        check_sh(I) <= '0';
                    end if;
                    
                    if ( wait_cnt_sh(I) = TIMEOUT ) then
                        error_sh(I) <= '1';
                    end if;
                    
                    if ( i_rdata(I)(31 downto 26) = "111111" and i_rdata(I)(37 downto 36) = "00"  and check_sh(I) = '1' ) then
                        check_sh(I) <= i_rempty(I);
                        sheader_time(I) <= i_rdata(I)(25 downto 20);
                        shtime <= i_rdata(I)(25 downto 20);
                        rack(I) <= not i_rempty(I);
                    end if;
                    
                    if ( check_sh(I) = '1' ) then
                        wait_cnt_sh(I) <= wait_cnt_sh(I) + '1';
                        rack(I) <= not i_rempty(I);
                    end if;
                END LOOP;
                
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
                    -- send sub header time -- check later if equal
                    o_wdata(25 downto 20) <= shtime;
                    o_we <= '1';
                end if;
                
            when merge_hits =>
                if ( error_shtime = '1' ) then
                    merge_state <= error_state;
                end if;
                
--                 compare 1/4
--                 min_value1 := "11111";
--                 FOR I in N/4 - 1 downto 0 LOOP
--                     check masking
--                     if ( i_mask_n(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                         cnt_trailer(I) <= '0';
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     check package start stop
--                     if ( i_rdata(I)(37 downto 36) = "01" or cnt_pre_header(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                     end if;
--                     
--                     if ( i_rdata(I)(37 downto 36) = "10" or cnt_trailer(I) = '0' ) then
--                         cnt_trailer(I) <= '0';
--                     end if;
--                                         
--                     if ( i_rdata(I)(31 downto 26) = "111111" or cnt_sh_header(I) = '0' ) then
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     find min value
--                     if ( 
--                         i_rdata(I)(37 downto 36) = "00" and 
--                         i_rdata(I)(31 downto 26) /= "111111" and 
--                         i_rempty(I) = '0' and
--                         i_mask_n(I) = '1' and
--                         i_wfull = '0' and
--                         w_ack = '0' and
--                         i_rdata(I)(35 downto 32) < min_value1
--                     ) then
--                         min_value1(3 downto 0) := i_rdata(I)(35 downto 32);
--                         min_value1(4) := '0';
--                         min_index1 := I;
--                     end if;
--                 END LOOP;
--                 
--                 compare 2/4
--                 min_value2 := "11111";
--                 FOR I in N/2 - 1 downto N/4 LOOP
--                     check masking
--                     if ( i_mask_n(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                         cnt_trailer(I) <= '0';
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     check package start stop
--                     if ( i_rdata(I)(37 downto 36) = "01" or cnt_pre_header(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                     end if;
--                     
--                     if ( i_rdata(I)(37 downto 36) = "10" or cnt_trailer(I) = '0' ) then
--                         cnt_trailer(I) <= '0';
--                     end if;
--                                         
--                     if ( i_rdata(I)(31 downto 26) = "111111" or cnt_sh_header(I) = '0' ) then
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     find min value
--                     if ( 
--                         i_rdata(I)(37 downto 36) = "00" and 
--                         i_rdata(I)(31 downto 26) /= "111111" and 
--                         i_rempty(I) = '0' and
--                         i_mask_n(I) = '1' and
--                         i_wfull = '0' and
--                         w_ack = '0' and
--                         i_rdata(I)(35 downto 32) < min_value2
--                     ) then
--                         min_value2(3 downto 0) := i_rdata(I)(35 downto 32);
--                         min_value2(4) := '0';
--                         min_index2 := I;
--                     end if;
--                 END LOOP;
--                 
--                 compare 3/4
--                 min_value3 := "11111";
--                 FOR I in N*3/4 - 1 downto N/2 LOOP
--                     check masking
--                     if ( i_mask_n(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                         cnt_trailer(I) <= '0';
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     check package start stop
--                     if ( i_rdata(I)(37 downto 36) = "01" or cnt_pre_header(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                     end if;
--                     
--                     if ( i_rdata(I)(37 downto 36) = "10" or cnt_trailer(I) = '0' ) then
--                         cnt_trailer(I) <= '0';
--                     end if;
--                                         
--                     if ( i_rdata(I)(31 downto 26) = "111111" or cnt_sh_header(I) = '0' ) then
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     find min value
--                     if ( 
--                         i_rdata(I)(37 downto 36) = "00" and 
--                         i_rdata(I)(31 downto 26) /= "111111" and 
--                         i_rempty(I) = '0' and
--                         i_mask_n(I) = '1' and
--                         i_wfull = '0' and
--                         w_ack = '0' and
--                         i_rdata(I)(35 downto 32) < min_value3
--                     ) then
--                         min_value3(3 downto 0) := i_rdata(I)(35 downto 32);
--                         min_value3(4) := '0';
--                         min_index3 := I;
--                     end if;
--                 END LOOP;
--                 
--                 compare 4/4
--                 min_value4 := "11111";
--                 FOR I in N - 1 downto N*3/4 LOOP
--                     check masking
--                     if ( i_mask_n(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                         cnt_trailer(I) <= '0';
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     check package start stop
--                     if ( i_rdata(I)(37 downto 36) = "01" or cnt_pre_header(I) = '0' ) then
--                         cnt_pre_header(I) <= '0';
--                     end if;
--                     
--                     if ( i_rdata(I)(37 downto 36) = "10" or cnt_trailer(I) = '0' ) then
--                         cnt_trailer(I) <= '0';
--                     end if;
--                                         
--                     if ( i_rdata(I)(31 downto 26) = "111111" or cnt_sh_header(I) = '0' ) then
--                         cnt_sh_header(I) <= '0';
--                     end if;
--                     
--                     find min value
--                     if ( 
--                         i_rdata(I)(37 downto 36) = "00" and 
--                         i_rdata(I)(31 downto 26) /= "111111" and 
--                         i_rempty(I) = '0' and
--                         i_mask_n(I) = '1' and
--                         i_wfull = '0' and
--                         w_ack = '0' and
--                         i_rdata(I)(35 downto 32) < min_value4
--                     ) then
--                         min_value4(3 downto 0) := i_rdata(I)(35 downto 32);
--                         min_value4(4) := '0';
--                         min_index4 := I;
--                     end if;
--                 END LOOP;

                FOR I in N - 1 downto 0 LOOP
                    -- check masking
                    if ( i_mask_n(I) = '0' ) then
                        cnt_pre_header(I) <= '0';
                        cnt_trailer(I) <= '0';
                        cnt_sh_header(I) <= '0';
                    end if;
                    
                    -- check package start stop
                    if ( i_rdata(I)(37 downto 36) = "01" or cnt_pre_header(I) = '0' ) then
                        cnt_pre_header(I) <= '0';
                    end if;
                    
                    if ( i_rdata(I)(37 downto 36) = "10" or cnt_trailer(I) = '0' ) then
                        cnt_trailer(I) <= '0';
                    end if;
                                        
                    if ( i_rdata(I)(31 downto 26) = "111111" or cnt_sh_header(I) = '0' ) then
                        cnt_sh_header(I) <= '0';
                    end if;
                END LOOP;
                
                -- find min values
                min_index1 := get_min_index(N/4, 0, link_good, "11111", i_rdata);
                min_index2 := get_min_index(N/2, N/4, link_good, "11111", i_rdata);
                min_index3 := get_min_index(N*3/4, N/2, link_good, "11111", i_rdata);
                min_index4 := get_min_index(N, N*3/4, link_good, "11111", i_rdata);
                
                min_value1 := get_hit_time(min_index1, i_rdata);
                min_value2 := get_hit_time(min_index2, i_rdata);
                min_value3 := get_hit_time(min_index3, i_rdata);
                min_value4 := get_hit_time(min_index4, i_rdata);
                
                w_ack <= '0';
                if ( min_value1 /= "11111" or min_value2 /= "11111" or min_value3 /= "11111" or min_value4 /= "11111" ) then
                    w_ack <= '1';
                    if ( min_value1 <= min_value2 and min_value1 <= min_value3 and min_value1 <= min_value4 ) then
                        min_hit <= i_rdata(min_index1);
                        rack(min_index1) <= '1';
                    elsif ( min_value2 <= min_value1 and min_value2 <= min_value3 and min_value2 <= min_value4 ) then
                        min_hit <= i_rdata(min_index2);
                        rack(min_index2) <= '1';
                    elsif ( min_value3 <= min_value1 and min_value3 <= min_value2 and min_value3 <= min_value4 ) then
                        min_hit <= i_rdata(min_index3);
                        rack(min_index3) <= '1';
                    elsif ( min_value4 <= min_value1 and min_value4 <= min_value3 and min_value4 <= min_value3 ) then
                        min_hit <= i_rdata(min_index4);
                        rack(min_index4) <= '1';
                    end if;
                end if;
                
--         N : integer;
--         Npart : integer;
--         empty : std_logic_vector;
--         minval : std_logic_vector;
--         wfull : std_logic;
--         wack : std_logic;
--         mask_n : std_logic_vector;
--         data : data_array(N - 1 downto 0)--;
                
                -- wait one cycle bcz of fifo
--                 w_ack <= '0';
--                 if ( min_value1 /= "11111" or min_value2 /= "11111" or min_value3 /= "11111" or min_value4 /= "11111" ) then
--                     w_ack <= '1';
--                     if ( min_value1 <= min_value2 and min_value1 <= min_value3 and min_value1 <= min_value4 ) then
--                         min_hit <= i_rdata(min_index1);
--                         rack(min_index1) <= '1';
--                     elsif ( min_value2 <= min_value1 and min_value2 <= min_value3 and min_value2 <= min_value4 ) then
--                         min_hit <= i_rdata(min_index2);
--                         rack(min_index2) <= '1';
--                     elsif ( min_value3 <= min_value1 and min_value3 <= min_value2 and min_value3 <= min_value4 ) then
--                         min_hit <= i_rdata(min_index3);
--                         rack(min_index3) <= '1';
--                     elsif ( min_value4 <= min_value1 and min_value4 <= min_value3 and min_value4 <= min_value3 ) then
--                         min_hit <= i_rdata(min_index4);
--                         rack(min_index4) <= '1';
--                     end if;
--                 end if;
                
                if ( w_ack = '1' ) then
                    o_wdata <= min_hit;
                    o_we <= '1';
                end if;
                
                if ( cnt_sh_header = check_zeros ) then
                    merge_state <= wait_for_sh;
                    cnt_sh_header <= (others => '1');
                end if;
                
                if ( cnt_pre_header = check_zeros ) then
                    merge_state <= wait_for_pre;
                    cnt_pre_header <= (others => '1');
                end if;
                
                if ( cnt_trailer = check_zeros ) then
                    merge_state <= trailer;
                    cnt_trailer <= (others => '1');
                end if;
                
            when trailer =>
                -- check for trailer
                FOR I in N - 1 downto 0 LOOP
                    if ( i_mask_n(I) = '0' ) then
                        check_tr(I) <= '0';
                    end if;
                
                    if ( wait_cnt_tr(I) = TIMEOUT ) then
                        error_tr(I) <= '1';
                    end if;
                    
                    if ( i_rempty(I) = '0' and i_rdata(I)(37 downto 36) = "10" and check_tr(I) = '1' ) then
                        check_tr(I) <= i_rempty(I); 
                        rack(I) <= not i_rempty(I); 
                    end if;
                    
                    if ( check_tr(I) = '1' ) then
                        wait_cnt_tr(I) <= wait_cnt_tr(I) + '1';
                        rack(I) <= not i_rempty(I); 
                    end if;
                END LOOP;
                
                -- check if fifo is not full and all links have subheader
                if( check_tr = check_zeros and i_wfull = '0' ) then
                    merge_state <= wait_for_pre;
                    -- reset signals
                    wait_cnt_tr <= (others => (others => '0'));
                    check_tr <= (others => '1');
                    -- send trailer
                    o_wdata(37 downto 36) <= "10";
                    o_wdata(11 downto 4) <= x"9C";
                    o_wdata(3 downto 0) <= "0000";
                    o_weop <= '1';
                    o_we <= '1';
                end if;
                                
            when error_state =>
                -- send error message xxxxxxDC
                -- 12: error gtime1
                -- 13: error gtime2
                -- 14: error shtime
                -- N+14 downto 14: error wait for pre
                o_wdata(3 downto 0) <= "0001";
                o_wdata(11 downto 4) <= x"DC";
                o_wdata(12) <= error_gtime1;
                o_wdata(13) <= error_gtime2;
                o_wdata(14) <= error_shtime;
                o_wdata(N + 14 downto 15) <= error_pre;
                o_weop <= '1';
                o_we <= '1';
                
            when others =>
                merge_state <= wait_for_pre;
                
        end case;
        --
    end if;
    end process;

end architecture;
