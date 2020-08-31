library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.dataflow_components.all;

-- merge packets delimited by SOP and EOP from N input streams
entity time_merger is
generic (
    W : positive := 32;
    TIMEOUT : std_logic_vector(31 downto 0) := x"FFFFFFFF";
    N : positive := 34--;
);
port (
    -- input streams
    i_rdata     : in    data_array(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0); -- start of packet (SOP)
    i_reop      : in    std_logic_vector(N-1 downto 0); -- end of packet (EOP)
    i_rshop     : in    std_logic_vector(N-1 downto 0); -- sub header of packet (SHOP)
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    i_mask_n    : in    std_logic_vector(N-1 downto 0);
    i_link      : in    integer;
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
 
    type fpga_id_array_t is array (N - 1 downto 0) of std_logic_vector(15 downto 0);
    type sheader_time_array_t is array (N - 1 downto 0) of std_logic_vector(5 downto 0);
    type merge_state_type is (wait_for_pre, compare_time1, compare_time2, wait_for_sh, error_state, merge_hits, get_time1, get_time2, trailer);
    subtype index_int is natural range 0 to 36; -- since we have a maximum number of 36 links, default is 36
    
    constant check_zeros : std_logic_vector(N - 1 downto 0) := (others => '0');
    constant check_ones : std_logic_vector(N - 1 downto 0) := (others => '1');
    constant check_zeros_t_3 : std_logic_vector(4 downto 0) := (others => '0');
    constant check_ones_t_3 : std_logic_vector(4 downto 0) := (others => '1');

    signal error_gtime1, error_gtime2, error_shtime, w_ack, error_merger, check_overflow : std_logic;
    signal merge_state : merge_state_type;
    signal rack, rack_hit, error_pre, error_sh, error_tr, sh_state, pre_state, tr_state : std_logic_vector(N - 1 downto 0);
    signal wait_cnt_pre, wait_cnt_sh, wait_cnt_merger : std_logic_vector(31 downto 0);
    signal gtime1, gtime2 : data_array(N - 1 downto 0);
    signal shtime : std_logic_vector(5 downto 0);
    signal overflow : std_logic_vector(15 downto 0);
    signal sheader_time : sheader_time_array_t;
    signal fpga_id : fpga_id_array_t;
    
    -- merge signals
    signal min_hit : std_logic_vector(37 downto 0);
    signal min_fpga_id : std_logic_vector(15 downto 0);
    signal link_good, sop_wait, shop_wait, time_wait, rack_link : std_logic_vector(N - 1 downto 0);
    
    -- layer zero signals
    type hit_t_0_array_t is array (N - 1 downto 0) of std_logic_vector(37 downto 0);
    signal data_t_0 : hit_t_0_array_t;
    signal full_t_0, empty_t_0 : std_logic_vector(N - 1 downto 0);
    
    -- layer one signals
    type hit_t_1_array_t is array (16 downto 0) of std_logic_vector(37 downto 0);
    signal data_t_1 : hit_t_1_array_t;
    signal full_t_1, empty_t_1 : std_logic_vector(16 downto 0);
    
    -- layer two signals
    type hit_t_2_array_t is array (8 downto 0) of std_logic_vector(37 downto 0);
    signal data_t_2 : hit_t_2_array_t;
    signal full_t_2, empty_t_2 : std_logic_vector(8 downto 0);
    
    -- layer three signals
    type hit_t_3_array_t is array (4 downto 0) of std_logic_vector(37 downto 0);
    signal data_t_3 : hit_t_3_array_t;
    signal full_t_3, empty_t_3 : std_logic_vector(4 downto 0);
        
begin

    -- error signals
    o_error_gtime(0) <= error_gtime1;
    o_error_gtime(1) <= error_gtime2;
    o_error_shtime <= error_shtime;
    o_error_pre <= error_pre;
    o_error_sh <= error_sh;
    
    generate_rack : FOR I in N-1 downto 0 GENERATE
        o_rack(I) <= rack(I) or rack_hit(I) or rack_link(I);
    END GENERATE;
    
    -- readout fifo
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        rack_link <= (others => '0');
        link_good <= (others => '0');
        sop_wait <= (others => '1');
        shop_wait <= (others => '1');
        time_wait <= (others => '1');
        sh_state <= (others => '1');
        pre_state <= (others => '1');
        tr_state <= (others => '1');
        fpga_id <= (others => (others => '0'));
        --
    elsif rising_edge(i_clk) then
    
        rack_link <= (others => '0');
    
        FOR I in N - 1 downto 0 LOOP
            -- link is good ('1') if link is not empty, wfull not full, not masked, w_ack is zero, i_rdata has hit data else '0'
            if ( i_rempty(I) = '0' and i_wfull = '0' and i_mask_n(I) = '1' and rack_hit(I) = '0' and i_rdata(I)(37 downto 36) = "00" and i_rdata(I)(31 downto 26) /= "111111" ) then
                link_good(I) <= '1';
            else
                link_good(I) <= '0';
            end if;
            
            -- read out fifo if not empty, not start of package and not masked
            if ( i_mask_n(I) = '0' ) then
                sop_wait(I) <= '0';
            elsif ( i_rempty(I) = '0' and i_rsop(I) = '1' ) then
                sop_wait(I) <= '0';
                fpga_id(I) <= i_rdata(I)(27 downto 12);
            elsif ( merge_state = wait_for_pre and rack_link(I) = '0' and i_rempty(I) = '0' ) then
                sop_wait(I) <= '1';
                rack_link(I) <= '1';
            end if;
            
            -- read out fifo if not empty, not sub header of package and not masked
            if ( i_mask_n(I) = '0' ) then
                shop_wait(I) <= '0';
            elsif ( i_rempty(I) = '0' and i_rshop(I) = '1' ) then
                shop_wait(I) <= '0';
            elsif ( merge_state = wait_for_sh and rack_link(I) = '0' and i_rempty(I) = '0' ) then
                shop_wait(I) <= '1';
                rack_link(I) <= '1';
            end if;

            -- check for time wait
            if ( i_rempty(I) = '0' and i_mask_n(I) = '1' and rack(I) = '0' and ( merge_state = get_time1 or merge_state = get_time2 ) ) then
                time_wait(I) <= '0';
            else
                time_wait(I) <= '1';
            end if;
            
            -- check for state change in merge_hits state
            if ( i_rempty(I) = '0' and i_rshop(I) = '1' and i_mask_n(I) = '1' and rack(I) = '0' and empty_t_3 = check_ones_t_3 and merge_state =  merge_hits ) then
                sh_state(I) <= '0';
            else
                sh_state(I) <= '1';
            end if;
            
            if ( i_rempty(I) = '0' and i_rsop(I) = '1' and i_mask_n(I) = '1' and rack(I) = '0' and empty_t_3 = check_ones_t_3 and merge_state =  merge_hits ) then
                pre_state(I) <= '0';
            else
                pre_state(I) <= '1';
            end if;
            
            if ( i_rempty(I) = '0' and i_reop(I) = '1' and i_mask_n(I) = '1' and rack(I) = '0' and empty_t_3 = check_ones_t_3 and merge_state =  merge_hits ) then
                tr_state(I) <= '0';
            else
                tr_state(I) <= '1';
            end if;
        END LOOP;
    end if;
    end process;
    
    -- merge hits
    process(i_clk, i_reset_n)
        variable min_value : std_logic_vector(3 downto 0);
        variable min_index : index_int;
        --
    begin
    if ( i_reset_n /= '1' ) then
        w_ack <= '0';
        min_hit <= (others => '1');
        min_fpga_id <= (others => '0');
        rack_hit <= (others => '0');
        
        data_t_0 <= (others => (others => '0'));
        full_t_0 <= (others => '0');
        empty_t_0 <= (others => '1');
        
        data_t_1 <= (others => (others => '0'));
        full_t_1 <= (others => '0');
        empty_t_1 <= (others => '1');
        
        data_t_2 <= (others => (others => '0'));
        full_t_2 <= (others => '0');
        empty_t_2 <= (others => '1');
        
        data_t_3 <= (others => (others => '0'));
        full_t_3 <= (others => '0'); 
        empty_t_3 <= (others => '1');
        --
    elsif rising_edge(i_clk) then
        rack_hit <= (others => '0');
        -- merge hits
        if ( merge_state = merge_hits ) then
        
            -- write to layer zero
            FOR I in N - 1 downto 0 LOOP
                if ( link_good(I) = '1' and empty_t_0(I) = '1' and full_t_0(I) = '0' ) then
                    data_t_0(I) <= i_rdata(I);
                    rack_hit(I) <= '1';
                    full_t_0(I) <= '1';
                    empty_t_0(I) <= '0';
                end if;
            END LOOP;

            -- write to layer one
            FOR I in 16 downto 0 LOOP
                if ( empty_t_0(I) = '0' and empty_t_1(I) = '1' and full_t_1(I) = '0' and data_t_0(I)(35 downto 32) <= data_t_0(I+17)(35 downto 32) ) then
                    data_t_1(I) <= data_t_0(I);
                    full_t_1(I) <= '1';
                    full_t_0(I) <= '0';
                    empty_t_0(I) <= '1';
                    empty_t_1(I) <= '0';
                elsif ( empty_t_0(I+17) = '0' and empty_t_1(I) = '1' and full_t_1(I) = '0' ) then
                    data_t_1(I) <= data_t_0(I+17);
                    full_t_1(I) <= '1';
                    full_t_0(I+17) <= '0';
                    empty_t_0(I+17) <= '1';
                    empty_t_1(I) <= '0';
                end if;
            END LOOP;
            
            -- write to layer two
            FOR I in 8 downto 0 LOOP
                if ( I = 0 ) then
                    if ( empty_t_1(I) = '0' and empty_t_2(I) = '1' and full_t_2(I) = '0' ) then
                        data_t_2(I) <= data_t_1(I);
                        full_t_2(I) <= '1';
                        full_t_1(I) <= '0';
                        empty_t_1(I) <= '1';
                        empty_t_2(I) <= '0';
                    end if;
                elsif ( empty_t_1(I) = '0' and empty_t_2(I) = '1' and full_t_2(I) = '0' and data_t_1(I)(35 downto 32) <= data_t_1(I+8)(35 downto 32) ) then
                    data_t_2(I) <= data_t_1(I);
                    full_t_2(I) <= '1';
                    full_t_1(I) <= '0';
                    empty_t_1(I) <= '1';
                    empty_t_2(I) <= '0';
                elsif ( empty_t_1(I+8) = '0' and empty_t_2(I) = '1' and full_t_2(I) = '0' ) then
                    data_t_2(I) <= data_t_1(I+8);
                    full_t_2(I) <= '1';
                    full_t_1(I+8) <= '0';
                    empty_t_1(I+8) <= '1';
                    empty_t_2(I) <= '0';
                end if;
            END LOOP;
            
            -- write to layer three
            FOR I in 4 downto 0 LOOP
                if ( I = 0 ) then
                    if ( empty_t_2(I) = '0' and empty_t_3(I) = '1' and full_t_3(I) = '0' ) then
                        data_t_3(I) <= data_t_2(I);
                        full_t_3(I) <= '1';
                        full_t_2(I) <= '0';
                        empty_t_2(I) <= '1';
                        empty_t_3(I) <= '0';
                    end if;
                elsif ( empty_t_2(I) = '0' and empty_t_3(I) = '1' and full_t_3(I) = '0' and data_t_2(I)(35 downto 32) <= data_t_2(I+4)(35 downto 32) ) then
                    data_t_3(I) <= data_t_2(I);
                    full_t_3(I) <= '1';
                    full_t_2(I) <= '0';
                    empty_t_2(I) <= '1';
                    empty_t_3(I) <= '0';
                elsif ( empty_t_2(I+4) = '0' and empty_t_3(I) = '1' and full_t_3(I) = '0' ) then
                    data_t_3(I) <= data_t_2(I+4);
                    full_t_3(I) <= '1';
                    full_t_2(I+4) <= '0';
                    empty_t_2(I+4) <= '1';
                    empty_t_3(I) <= '0';
                end if;
            END LOOP;
            
            -- read out last layer
            if ( empty_t_3 /= check_zeros_t_3 ) then
                min_value := min_hit(35 downto 32);
            else
                min_value := "1111";
            end if;
            min_index := 36;
            FOR I in 4 downto 0 LOOP
                if ( empty_t_3(I) = '0' and i_wfull = '0' and data_t_3(I)(35 downto 32) <= min_value ) then
                    min_value := data_t_3(I)(35 downto 32);
                    min_index := I;
                end if;
            END LOOP;
            
            w_ack <= '0';
            if ( min_index /= 36 ) then
                w_ack <= '1';
                min_hit <= data_t_3(min_index);
                min_fpga_id <= fpga_id(min_index);
                full_t_3(min_index) <= '0';
                empty_t_3(min_index) <= '1';
            end if;
        else
            min_hit <= (others => '1');
        end if;
    end if;
    end process;
    
    -- write data
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        merge_state <= wait_for_pre;
        error_pre <= (others => '0');
        error_sh <= (others => '0');
        error_tr <= (others => '0');
        wait_cnt_pre <= (others => '0');
        wait_cnt_sh <= (others => '0');
        wait_cnt_merger <= (others => '0');
        gtime1 <= (others => (others => '0'));
        gtime2 <= (others => (others => '0'));
        shtime <= (others => '1');
        sheader_time <= (others => (others => '0'));
        error_gtime1 <= '0';
        error_gtime2 <= '0';
        error_shtime <= '0';
        error_merger <= '0';
        o_wdata <= (others => '0');
        overflow <= (others => '0');
        rack <= (others => '0');
        o_wsop <= '0';
        o_weop <= '0';
        o_we <= '0';
        check_overflow <= '0';
        --
    elsif rising_edge(i_clk) then
        
        rack <= (others => '0');
        o_we <= '0';
        o_wsop <= '0';
        o_weop <= '0';
    
        case merge_state is
            when wait_for_pre =>
                -- readout until all fifos have preamble
                if ( sop_wait /= check_zeros ) then
                    wait_cnt_pre <= wait_cnt_pre + '1';
                elsif ( i_wfull = '0' ) then
                    merge_state <= get_time1;
                    rack <= (others => '1');
                    -- reset signals
                    wait_cnt_pre <= (others => '0');
                    -- send merged data preamble
                    -- sop & preamble & zeros & datak
                    o_wdata(33 downto 32) <= "01";
                    o_wdata(31 downto 26) <= "111010";
                    o_wdata(7 downto 0) <= x"BC";
                    o_we <= '1';
                    o_wsop <= '1';
                end if;
                
                -- if wait for pre gets timeout
                if ( wait_cnt_pre = TIMEOUT ) then
                    error_pre <= sop_wait;
                    merge_state <= error_state;
                end if;
                
            when get_time1 =>
                -- get MSB from FPGA time
                if ( time_wait = check_zeros ) then
                    merge_state <= compare_time1;
                    gtime1 <= i_rdata;
                end if;
                
            when compare_time1 =>
                -- compare MSB from FPGA time
                FOR I in N - 1 downto 0 LOOP
                    if ( gtime1(I) /= gtime1(i_link) and i_mask_n(I) = '1' ) then
                        error_gtime1 <= '1';
                    end if;
                END LOOP;
                
                -- check if fifo is not full and all links have same time              
                if ( error_gtime1 = '0' and i_wfull = '0' ) then
                    merge_state <= get_time2;
                    rack <= (others => '1');
                    -- reset signals
                    gtime1 <= (others => (others => '0'));
                    -- send gtime1
                    o_wdata(67 downto 66) <= "00";
                    o_wdata(65 downto 34) <= gtime1(i_link)(35 downto 4);
                end if;
                
            when get_time2 =>
                -- get LSB from FPGA time
                if ( error_gtime1 = '1' ) then
                    merge_state <= error_state;
                elsif ( time_wait = check_zeros ) then
                    merge_state <= compare_time2;
                    gtime2 <= i_rdata;
                end if;
                
            when compare_time2 =>
                -- compare LSB from FPGA time
                FOR I in N - 1 downto 0 LOOP
                    if ( gtime2(I) /= gtime2(i_link) and i_mask_n(I) = '1' ) then
                        error_gtime2 <= '1';
                    end if;
                END LOOP;
                
                -- check if fifo is not full and all links have same time
                if ( error_gtime2 = '0' and i_wfull = '0' ) then
                    merge_state <= wait_for_sh;
                    -- reset signals
                    gtime2 <= (others => (others => '0'));
                    -- send gtime2
                    o_wdata(33 downto 32) <= "00";
                    o_wdata(31 downto 0) <= gtime2(i_link)(35 downto 4);
                    o_we <= '1';
                end if;
                
            when wait_for_sh =>
                if ( error_gtime2 = '1' ) then
                    merge_state <= error_state;
                end if;
            
                -- readout until all fifos have sub header
                if ( shop_wait /= check_zeros ) then
                    wait_cnt_sh <= wait_cnt_sh + '1';
                -- TODO handle overflow
--                 elsif ( check_overflow = '1' ) then    
--                     check_overflow <= '0';
--                     FOR I in 15 downto 0 LOOP
--                         if ( i_rdata(N-1 downto 0)(I + 4) = 0 ) then
--                             overflow(I) <= '0';
--                         else
--                             overflow(I) <= '1';
--                         end if;
--                     END LOOP;
                elsif ( i_wfull = '0' ) then
                    merge_state <= merge_hits;
                    rack <= (others => '1');
                    -- reset signals
                    wait_cnt_sh <= (others => '0');
                    wait_cnt_merger <= (others => '0');
                    overflow <= (others => '0');
                    -- send merged data sub header
                    -- zeros & sub header & zeros & datak
                    o_wdata(33 downto 32) <= "11";
                    o_wdata(31 downto 28) <= "0000";
                    o_wdata(27 downto 22) <= "111111";
                    -- send sub header time -- check later if equal
                    o_wdata(21 downto 16) <= i_rdata(i_link)(25 downto 20);
                    shtime <= i_rdata(i_link)(25 downto 20);
                    FOR I in N - 1 downto 0 LOOP
                        if ( i_mask_n(I) = '1' ) then
                            sheader_time(I) <= i_rdata(I)(25 downto 20);
                        end if;
                    END LOOP;
                    o_wdata(15 downto 0) <= overflow;
                    o_we <= '1';
                end if;
                
                -- if wait for pre gets timeout
                if ( wait_cnt_sh = TIMEOUT ) then
                    error_sh <= shop_wait;
                    merge_state <= error_state;
                end if;
                
            when merge_hits =>
                if ( error_shtime = '1' ) then
                    merge_state <= error_state;
                end if;

                -- check if sheader time is equal
                FOR I in N - 1 downto 0 LOOP
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
                if ( sh_state = check_zeros ) then
                    merge_state <= wait_for_sh;
                end if;
                
                -- TODO error if pre is not there
                if ( pre_state = check_zeros ) then
                    merge_state <= wait_for_pre;
                end if;
                
                -- TODO error if trailer is not there
                if ( tr_state = check_zeros ) then
                    merge_state <= trailer;
                end if;
                
                -- write out data
                if ( w_ack = '1' ) then
                    o_wdata(49 downto 34) <= min_fpga_id;
                    o_wdata(33 downto 0) <= min_hit(37 downto 4);
                    o_we <= '1';
                end if;
                
            when trailer =>
                -- send trailer
                if( i_wfull = '0' ) then
                    merge_state <= wait_for_pre;
                    -- send trailer
                    o_wdata(37 downto 36) <= "10";
                    o_wdata(35 downto 12) <= (others => '0');
                    o_wdata(11 downto 4) <= x"9C";
                    o_wdata(3 downto 0) <= "0001";
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
                o_wdata(15) <= error_merger;
                if ( error_pre /= check_zeros ) then
                    o_wdata(16) <= '1';
                end if;
                if ( error_sh /= check_zeros ) then
                    o_wdata(17) <= '1';
                end if;
                o_weop <= '1';
                o_we <= '1';
                
            when others =>
                merge_state <= wait_for_pre;
                
        end case;
        --
    end if;
    end process;

end architecture;
