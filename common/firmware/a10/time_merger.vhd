library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;
use work.dataflow_components.all;
--use std.textio.all;

-- merge packets delimited by SOP and EOP from N input streams
entity time_merger is
generic (
    W : positive := 66+12;
    TIMEOUT : std_logic_vector(31 downto 0) := x"FFFFFFFF";
    TREE_DEPTH_w : positive := 8;
    TREE_DEPTH_r : positive := 8;
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
    o_rdata     : out   std_logic_vector(W-1 downto 0);
    i_ren       : in    std_logic;
    o_empty     : out   std_logic;

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
    subtype index_int is natural range 0 to 36; -- since we have a maximum number of 36 links, default is 36
    
    constant check_zeros : std_logic_vector(N - 1 downto 0) := (others => '0');
    constant check_ones : std_logic_vector(N - 1 downto 0) := (others => '1');
    constant check_zeros_t_3 : std_logic_vector(4 downto 0) := (others => '0');
    constant check_ones_t_3 : std_logic_vector(4 downto 0) := (others => '1');

    signal error_gtime1, error_gtime2, error_shtime, error_merger, check_overflow, header_trailer_we : std_logic;
    type merge_state_type is (wait_for_pre, compare_time1, compare_time2, wait_for_sh, error_state, merge_hits, get_time1, get_time2, trailer, read_hits, wait_for_sh_written);
    signal merge_state : merge_state_type;
    signal merger_state_signal : std_logic;
    signal rack, rack_hit, error_pre, error_sh, error_tr, sh_state, pre_state, tr_state : std_logic_vector(N - 1 downto 0);
    signal wait_cnt_pre, wait_cnt_sh, wait_cnt_merger : std_logic_vector(31 downto 0);
    signal header_trailer : std_logic_vector(33 downto 0);
    signal gtime1, gtime2 : data_array(N - 1 downto 0);
    signal shtime : std_logic_vector(5 downto 0);
    signal overflow : std_logic_vector(15 downto 0);
    signal sheader_time : sheader_time_array_t;
    signal fpga_id : fpga_id_array_t;
    
    -- merge signals
    signal min_fpga_id : std_logic_vector(15 downto 0);
    signal sop_wait, shop_wait, time_wait, rack_link : std_logic_vector(N - 1 downto 0);
    signal link_good : std_logic_vector(63 downto 0);
    
    -- merger tree (at the moment for 32 links)
    type fifo_width_t is array (6 downto 0) of integer;
    constant read_width : fifo_width_t := (W, 64+12, 64+12, 64+12, 64+12, 64+12, 64+12);
    constant write_width : fifo_width_t := (64+2+12, 64+12, 64+12, 64+12, 64+12, 64+12, 32+6);
    constant generate_fifos : fifo_width_t := (1, 2, 4, 8, 16, 32, 64);
        
    signal fifo_data_0              : fifo_array_38(generate_fifos(0) - 1 downto 0);
    signal fifo_q_0, fifo_q_0_reg   : fifo_array_76(generate_fifos(0) - 1 downto 0);
    signal wait_cnt_fifo_0          : fifo_array_2(generate_fifos(0) - 1 downto 0);
    signal layer_0_state            : fifo_array_4(generate_fifos(0) - 1 downto 0);
    signal layer_0_cnt              : fifo_array_32(generate_fifos(0) - 1 downto 0);
    signal fifo_ren_0, fifo_ren_0_reg, fifo_wen_0, fifo_full_0, fifo_empty_0, fifo_empty_0_reg, saw_header_0, saw_trailer_0, reset_fifo_0 : std_logic_vector(generate_fifos(0) - 1 downto 0);

    signal fifo_q_1 : fifo_array_76(generate_fifos(1) - 1 downto 0);
    signal fifo_q_2 : fifo_array_76(generate_fifos(2) - 1 downto 0);
    signal fifo_q_3 : fifo_array_76(generate_fifos(3) - 1 downto 0);
    signal fifo_q_4 : fifo_array_76(generate_fifos(4) - 1 downto 0);
    signal fifo_q_5 : fifo_array_76(generate_fifos(5) - 1 downto 0);
    signal fifo_empty_1, fifo_ren_1 : std_logic_vector(generate_fifos(1) - 1 downto 0);
    signal fifo_empty_2, fifo_ren_2 : std_logic_vector(generate_fifos(2) - 1 downto 0);
    signal fifo_empty_3, fifo_ren_3 : std_logic_vector(generate_fifos(3) - 1 downto 0);
    signal fifo_empty_4, fifo_ren_4 : std_logic_vector(generate_fifos(4) - 1 downto 0);
    signal fifo_empty_5, fifo_ren_5, fifo_ren_5_reg : std_logic_vector(generate_fifos(5) - 1 downto 0);

    signal mask_n_1 : std_logic_vector(generate_fifos(1) - 1 downto 0);
    signal mask_n_2 : std_logic_vector(generate_fifos(2) - 1 downto 0);
    signal mask_n_3 : std_logic_vector(generate_fifos(3) - 1 downto 0);
    signal mask_n_4 : std_logic_vector(generate_fifos(4) - 1 downto 0);
    signal mask_n_5 : std_logic_vector(generate_fifos(5) - 1 downto 0);

    constant size_last : integer := generate_fifos(5)/2;
    signal fifo_data_6              : fifo_array_78(generate_fifos(6) - 1 downto 0);
    signal fifo_q_6, fifo_q_6_reg   : fifo_array_78(generate_fifos(6) - 1 downto 0);
    signal wait_cnt_fifo_6          : fifo_array_2(generate_fifos(6) - 1 downto 0);
    signal layer_6_state            : fifo_array_4(generate_fifos(6) - 1 downto 0);
    signal fifo_wen_6, fifo_full_6  : std_logic_vector(generate_fifos(6) - 1 downto 0);
    signal alignment_done : std_logic_vector(generate_fifos(6) - 1 downto 0) := (others => '0');
    
begin

    -- ports out
    o_error_gtime(0) <= error_gtime1;
    o_error_gtime(1) <= error_gtime2;
    o_error_shtime <= error_shtime;
    o_error_pre <= error_pre;
    o_error_sh <= error_sh;
    
    generate_rack : FOR I in N-1 downto 0 GENERATE
        o_rack(I) <= rack(I) or rack_hit(I) or rack_link(I);
    END GENERATE;
    
     -- generate tree fifos
    -- fix this now for 32 links
    fifos_first:
    FOR j in 0 to generate_fifos(0)-1 GENERATE
        e_link_fifo : entity work.ip_dcfifo_mixed_widths
        generic map(
            ADDR_WIDTH_w => TREE_DEPTH_w,
            DATA_WIDTH_w => write_width(0),
            ADDR_WIDTH_r => TREE_DEPTH_r,
            DATA_WIDTH_r => read_width(0),
            DEVICE 		 => "Arria 10"--,
        )
        port map (
            aclr 	=> not i_reset_n or reset_fifo_0(j),
            data 	=> fifo_data_0(j),
            rdclk 	=> i_clk,
            rdreq 	=> fifo_ren_0(j),
            wrclk 	=> i_clk,
            wrreq 	=> fifo_wen_0(j),
            q 		=> fifo_q_0_reg(j),
            rdempty => fifo_empty_0_reg(j),
            rdusedw => open,
            wrfull 	=> fifo_full_0(j),
            wrusedw => open--,
        );
    END GENERATE fifos_first;
    
    fifos_last:
    FOR j in 0 to generate_fifos(6)-1 GENERATE
        e_link_fifo : entity work.ip_dcfifo_mixed_widths
        generic map(
            ADDR_WIDTH_w => TREE_DEPTH_w,
            DATA_WIDTH_w => write_width(6),
            ADDR_WIDTH_r => TREE_DEPTH_r,
            DATA_WIDTH_r => read_width(6),
            DEVICE 		 => "Arria 10"--,
        )
        port map (
            aclr 	=> not i_reset_n,
            data 	=> fifo_data_6(j),
            rdclk 	=> i_clk,
            rdreq 	=> i_ren, --fifo_ren_2(j),
            wrclk 	=> i_clk,
            wrreq 	=> fifo_wen_6(j),
            q 		=> o_rdata,
            rdempty => o_empty,
            rdusedw => open,
            wrfull 	=> fifo_full_6(j),
            wrusedw => open--,
        );
    END GENERATE fifos_last;
    
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
            if ( i_rempty(I) = '0' and i_mask_n(I) = '1' and i_rdata(I)(37 downto 36) = "00" and i_rdata(I)(31 downto 26) /= "111111" ) then
                link_good(I) <= '1';
            else
                link_good(I) <= '0';
            end if;
            
            -- read out fifo if not empty, not start of package and not masked
            if ( merge_state /= wait_for_pre ) then
                sop_wait(I) <= '1';
            elsif ( i_mask_n(I) = '0' ) then
                sop_wait(I) <= '0';
            elsif ( i_rempty(I) = '0' and i_rsop(I) = '1' ) then
                sop_wait(I) <= '0';
                fpga_id(I) <= i_rdata(I)(27 downto 12);
            elsif ( merge_state = wait_for_pre and rack_link(I) = '0' and i_rempty(I) = '0' ) then
                sop_wait(I) <= '1';
                rack_link(I) <= '1';
            end if;
            
            -- read out fifo if not empty, not sub header of package and not masked
            if ( merge_state /= wait_for_sh ) then
                shop_wait(I) <= '1';
            elsif ( i_mask_n(I) = '0' ) then
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
            elsif ( i_mask_n(I) = '0' ) then
                time_wait(I) <= '0';
            else
                time_wait(I) <= '1';
            end if;
            
            -- check for state change in merge_hits state
            if ( i_rempty(I) = '0' and i_rshop(I) = '1' and i_mask_n(I) = '1' and rack(I) = '0' and rack_hit(I) = '0' and merge_state =  merge_hits ) then
                sh_state(I) <= '0';
            elsif ( i_mask_n(I) = '0' ) then
                sh_state(I) <= '0';
            else
                sh_state(I) <= '1';
            end if;
            
            if ( i_rempty(I) = '0' and i_rsop(I) = '1' and i_mask_n(I) = '1' and rack(I) = '0' and rack_hit(I) = '0' and merge_state =  merge_hits ) then
                pre_state(I) <= '0';
            elsif ( i_mask_n(I) = '0' ) then
                pre_state(I) <= '0';
            else
                pre_state(I) <= '1';
            end if;
            
            if ( i_rempty(I) = '0' and i_reop(I) = '1' and i_mask_n(I) = '1' and rack(I) = '0' and rack_hit(I) = '0' and merge_state =  merge_hits ) then
                tr_state(I) <= '0';
            elsif ( i_mask_n(I) = '0' ) then
                tr_state(I) <= '0';
            else
                tr_state(I) <= '1';
            end if;
        END LOOP;
    end if;
    end process;


    tree_layer_first:
    FOR i in 0 to generate_fifos(0) - 1 GENERATE
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        rack_hit(i) <= '0';
        fifo_data_0(i) <= (others => '0');
        fifo_wen_0(i) <= '0';
        saw_header_0(i) <= '0';
        saw_trailer_0(i) <= '0';
        layer_0_state(i) <= "0000";
        layer_0_cnt(i) <= (others => '0');
        reset_fifo_0(i) <= '0';
        wait_cnt_fifo_0(i) <= (others => '0');
        --
    elsif rising_edge(i_clk) then
        rack_hit(i) <= '0';
        fifo_wen_0(i) <= '0';
        reset_fifo_0(i) <= '0';
        if ( wait_cnt_fifo_0(i) /= "11" ) then
            wait_cnt_fifo_0(i) <= wait_cnt_fifo_0(i) + '1';
        end if;

        if ( merge_state = merge_hits ) then
            case layer_0_state(i) is
                
                when "0000" =>
                    if ( i_mask_n(i) = '0' ) then
                        saw_header_0(i) <= '1';
                        saw_trailer_0(i) <= '1';
                        layer_0_state(i) <= "0001";
                        fifo_data_0(i) <= tree_padding;
                        layer_0_cnt(i) <= layer_0_cnt(i) + '1';
                        fifo_wen_0(i) <= '1';
                    elsif ( fifo_full_0(i) = '1' or reset_fifo_0(i) = '1' or wait_cnt_fifo_0(i) /= "11" ) then
                        --
                    else
                        if ( fifo_full_0(i) = '0' and link_good(i) = '1' and i_rempty(i) = '0' and rack_hit(i) = '0' and i_rdata(i)(31 downto 26) /= "111111" and i_rdata(i)(37 downto 36) = "00" and reset_fifo_0(i) = '0' ) then
                            fifo_data_0(i) <= link_36_to_std(i) & i_rdata(i)(35 downto 4);
                            fifo_wen_0(i) <= '1';
                            rack_hit(i) <= '1';
                            layer_0_cnt(i) <= layer_0_cnt(i) + '1';
                            saw_header_0(i) <= '0';
                            saw_trailer_0(i) <= '0';
                        -- TODO: is this fine to quite until one is written (cnt > 0)?
                        elsif ( i_rdata(i)(31 downto 26) = "111111" and reset_fifo_0(i) = '0' ) then
                            saw_header_0(i) <= '1';
                            layer_0_state(i) <= "0001";
                            fifo_data_0(i) <= tree_padding;
                            layer_0_cnt(i) <= layer_0_cnt(i) + '1';
                            fifo_wen_0(i) <= '1';
                        -- TODO: is this fine to quite until one is written (cnt > 0)?
                        elsif ( i_rdata(i)(37 downto 36) /= "00" and reset_fifo_0(i) = '0' ) then
                            saw_trailer_0(i) <= '1';
                            layer_0_state(i) <= "0001";
                            fifo_data_0(i) <= tree_padding;
                            layer_0_cnt(i) <= layer_0_cnt(i) + '1';
                            fifo_wen_0(i) <= '1';
                        end if;
                    end if;
                when "0001" =>
                    if ( layer_0_cnt(i)(5) = '1' ) then
                        layer_0_state(i) <= "1111";
                    else
                        fifo_data_0(i) <= tree_padding;
                        layer_0_cnt(i) <= layer_0_cnt(i) + '1';
                        fifo_wen_0(i) <= '1';
                    end if;
                when "1111" =>
                    -- TODO: write out something?
                when others =>
                    layer_0_state(i) <= "0000";
            end case;
        else
            wait_cnt_fifo_0(i) <= (others => '0');
            reset_fifo_0(i) <= '1';
            layer_0_state(i) <= "0000";
            saw_header_0(i) <= '0';
            saw_trailer_0(i) <= '0';
            layer_0_cnt(i) <= (others => '0');
            fifo_data_0(i) <= (others => '0');
        end if;
    end if;
    end process;
    
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        fifo_q_0(i) <= (others => '0');
        fifo_empty_0(i) <= '0';
        --
    elsif rising_edge(i_clk) then
        fifo_q_0(i) <= fifo_q_0_reg(i);
        fifo_empty_0(i) <= fifo_empty_0_reg(i);
    end if;
    end process;
    END GENERATE tree_layer_first;

    merger_state_signal <= '1' when merge_state = merge_hits else '0';
    
    layer_1 : entity work.time_merger_tree_fifo_64
    generic map (  
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r,
        r_width => read_width(0), w_width => write_width(1),
        compare_fifos => generate_fifos(0), gen_fifos => generate_fifos(1)--,
    )
    port map (
        i_fifo_q        => fifo_q_0,
        i_fifo_empty    => fifo_empty_0,
        i_fifo_ren      => fifo_ren_1,
        i_merge_state   => merger_state_signal,
        i_mask_n        => i_mask_n,

        o_fifo_q        => fifo_q_1,
        o_fifo_empty    => fifo_empty_1,
        o_fifo_ren      => fifo_ren_0,
        o_mask_n        => mask_n_1,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_2 : entity work.time_merger_tree_fifo_64
    generic map (  
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r,
        r_width => read_width(1), w_width => write_width(2),
        compare_fifos => generate_fifos(1), gen_fifos => generate_fifos(2)--,
    )
    port map (
        i_fifo_q        => fifo_q_1,
        i_fifo_empty    => fifo_empty_1,
        i_fifo_ren      => fifo_ren_2,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_1,

        o_fifo_q        => fifo_q_2,
        o_fifo_empty    => fifo_empty_2,
        o_fifo_ren      => fifo_ren_1,
        o_mask_n        => mask_n_2,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_3 : entity work.time_merger_tree_fifo_64
    generic map (  
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r,
        r_width => read_width(2), w_width => write_width(3),
        compare_fifos => generate_fifos(2), gen_fifos => generate_fifos(3)--,
    )
    port map (
        i_fifo_q        => fifo_q_2,
        i_fifo_empty    => fifo_empty_2,
        i_fifo_ren      => fifo_ren_3,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_2,

        o_fifo_q        => fifo_q_3,
        o_fifo_empty    => fifo_empty_3,
        o_fifo_ren      => fifo_ren_2,
        o_mask_n        => mask_n_3,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_4 : entity work.time_merger_tree_fifo_64
    generic map (  
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r,
        r_width => read_width(3), w_width => write_width(4),
        compare_fifos => generate_fifos(3), gen_fifos => generate_fifos(4)--,
    )
    port map (
        i_fifo_q        => fifo_q_3,
        i_fifo_empty    => fifo_empty_3,
        i_fifo_ren      => fifo_ren_4,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_3,

        o_fifo_q        => fifo_q_4,
        o_fifo_empty    => fifo_empty_4,
        o_fifo_ren      => fifo_ren_3,
        o_mask_n        => mask_n_4,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    layer_5 : entity work.time_merger_tree_fifo_64
    generic map (  
        TREE_w => TREE_DEPTH_w, TREE_r => TREE_DEPTH_r,
        r_width => read_width(4), w_width => write_width(5),
        compare_fifos => generate_fifos(4), gen_fifos => generate_fifos(5)--,
    )
    port map (
        i_fifo_q        => fifo_q_4,
        i_fifo_empty    => fifo_empty_4,
        i_fifo_ren      => fifo_ren_5,
        i_merge_state   => merger_state_signal,
        i_mask_n        => mask_n_4,

        o_fifo_q        => fifo_q_5,
        o_fifo_empty    => fifo_empty_5,
        o_fifo_ren      => fifo_ren_4,
        o_mask_n        => mask_n_5,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );
    
    tree_layer_last:
    FOR i in 0 to generate_fifos(6) - 1 GENERATE
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        fifo_ren_5(i) <= '0';
        fifo_ren_5(i + size_last) <= '0';
        fifo_ren_5_reg(i) <= '0';
        fifo_ren_5_reg(i + size_last) <= '0';
        fifo_wen_6(i) <= '0';
        fifo_data_6(i) <= (others => '0');
        layer_6_state(i) <= (others => '0');
        alignment_done(i) <= '0';
        --
    elsif rising_edge(i_clk) then
        fifo_ren_5(i) <= '0';
        fifo_ren_5(i + size_last) <= '0';
        fifo_ren_5_reg(i) <= fifo_ren_5(i);
        fifo_ren_5_reg(i + size_last) <= fifo_ren_5(i + size_last);
        fifo_data_6(i)(W-1 downto W-2) <= "00";

        fifo_wen_6(i) <= '0';
        if ( merge_state = merge_hits and fifo_empty_5(i) = '0' and fifo_empty_5(i + size_last) = '0' ) then
            if ( fifo_data_6(i)(37 downto 0) = tree_padding and fifo_data_6(i)(75 downto 38) = tree_padding ) then
                alignment_done(i) <= '1';
            end if;
            if ( alignment_done(i) = '0' ) then
                case layer_6_state(i) is
                    when "0000" =>
                        if ( fifo_full_6(i) = '1' ) then
                            --
                        else
                            -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
                            if ( fifo_q_5(i)(31 downto 28) <= fifo_q_5(i + size_last)(31 downto 28) and fifo_empty_5(i) = '0' and fifo_ren_5(i) = '0' and fifo_ren_5_reg(i) = '0' ) then
                                fifo_data_6(i)(37 downto 0) <= fifo_q_5(i)(37 downto 0);
                                layer_6_state(i)(0) <= '1';
                                -- TODO: try to find out if tree_zero is not a valid hit
                                if ( fifo_q_5(i)(69 downto 66) <= fifo_q_5(i + size_last)(31 downto 28) and fifo_q_5(i)(75 downto 38) /= tree_zero ) then
                                    fifo_data_6(i)(75 downto 38) <= fifo_q_5(i)(75 downto 38);
                                    layer_6_state(i)(1) <= '1';
                                    fifo_wen_6(i) <= '1';
                                    fifo_ren_5(i) <= '1';
                                elsif ( fifo_q_5(i + size_last)(75 downto 38) /= tree_zero and fifo_empty_5(i + size_last) = '0' and fifo_ren_5(i + size_last) = '0' and fifo_ren_5_reg(i + size_last) = '0' ) then
                                    fifo_data_6(i)(75 downto 38) <= fifo_q_5(i + size_last)(37 downto 0);
                                    layer_6_state(i)(2) <= '1';
                                    fifo_wen_6(i) <= '1';
                                end if;
                            elsif ( fifo_empty_5(i + size_last) = '0' and fifo_ren_5(i + size_last) = '0' and fifo_ren_5_reg(i + size_last) = '0' ) then
                                fifo_data_6(i)(37 downto 0) <= fifo_q_5(i + size_last)(37 downto 0);
                                layer_6_state(i)(2) <= '1';
                                if ( fifo_q_5(i)(31 downto 28) <= fifo_q_5(i + size_last)(69 downto 66) and fifo_empty_5(i) = '0' and fifo_ren_5(i) = '0' and fifo_ren_5_reg(i) = '0' ) then
                                    fifo_data_6(i)(75 downto 38) <= fifo_q_5(i)(37 downto 0);
                                    layer_6_state(i)(0) <= '1';
                                    fifo_wen_6(i) <= '1';
                                elsif ( fifo_q_5(i + size_last)(75 downto 38) /= tree_zero ) then
                                    fifo_data_6(i)(75 downto 38) <= fifo_q_5(i + size_last)(75 downto 38);
                                    layer_6_state(i)(3) <= '1';
                                    fifo_wen_6(i) <= '1';
                                    fifo_ren_5(i + size_last) <= '1';
                                end if;
                            end if;
                        end if;
                    when "0011" =>
                        layer_6_state(i) <= (others => '0');
                        -- TODO: probably not needed
                        fifo_ren_5_reg(i) <= '1';
                        fifo_ren_5_reg(i + size_last) <= '1';
                    when "1100" =>
                        layer_6_state(i) <= (others => '0');
                        -- TODO: probably not needed
                        fifo_ren_5_reg(i) <= '1';
                        fifo_ren_5_reg(i + size_last) <= '1';
                    when "0101" =>
                        if ( fifo_q_5(i)(69 downto 66) <= fifo_q_5(i + size_last)(69 downto 66) and fifo_q_5(i)(75 downto 38) /= tree_zero ) then
                            fifo_data_6(i)(37 downto 0) <= fifo_q_5(i)(75 downto 38);
                            layer_6_state(i)(0) <= '0';
                            fifo_ren_5(i) <= '1';
                        elsif ( fifo_q_5(i + size_last)(75 downto 38) /= tree_zero ) then
                            fifo_data_6(i)(37 downto 0) <= fifo_q_5(i + size_last)(75 downto 38);
                            layer_6_state(i)(2) <= '0';
                            fifo_ren_5(i + size_last) <= '1';
                        end if;
                    when "0100" =>
                        -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
                        if ( fifo_empty_5(i) = '0' and fifo_ren_5(i) = '0' and fifo_ren_5_reg(i) = '0' ) then
                            -- TODO: what to do when fifo_q_1(i + size_last)(69 downto 66) is zero? maybe error cnt?
                            if ( fifo_q_5(i)(31 downto 28) <= fifo_q_5(i + size_last)(69 downto 66) ) then
                                fifo_data_6(i)(75 downto 38) <= fifo_q_5(i)(37 downto 0);
                                layer_6_state(i)(0) <= '1';
                                fifo_wen_6(i) <= '1';
                            elsif ( fifo_q_5(i + size_last)(75 downto 38) /= tree_zero ) then
                                fifo_data_6(i)(75 downto 38) <= fifo_q_5(i + size_last)(75 downto 38);
                                layer_6_state(i)(3) <= '1';
                                fifo_wen_6(i) <= '1';
                                fifo_ren_5(i + size_last) <= '1';
                            end if;
                        else
                            -- TODO: wait for fifo_0 i here --> error counter?
                        end if;
                    when "0001" =>
                        -- TODO: define signal for empty since the fifo should be able to get empty if no hits are comming
                        if ( fifo_empty_5(i + size_last) = '0' and fifo_ren_5(i + size_last) = '0' and fifo_ren_5_reg(i + size_last) = '0' ) then       
                            -- TODO: what to do when fifo_q_1(i)(69 downto 66) is zero? maybe error cnt?     
                            if ( fifo_q_5(i)(69 downto 66) <= fifo_q_5(i + size_last)(31 downto 28) and fifo_q_5(i)(75 downto 38) /= tree_zero ) then
                                fifo_data_6(i)(75 downto 38) <= fifo_q_5(i)(75 downto 38);
                                layer_6_state(i)(1) <= '1';
                                fifo_wen_6(i) <= '1';
                                fifo_ren_5(i) <= '1';
                            else
                                fifo_data_6(i)(75 downto 38) <= fifo_q_5(i + size_last)(37 downto 0);
                                layer_6_state(i)(2) <= '1';
                                fifo_wen_6(i) <= '1';
                            end if;
                        else
                            -- TODO: wait for fifo_0 i+size_last here --> error counter?
                        end if;
                    when others =>
                        layer_6_state(i) <= (others => '0');
                end case;
            end if;
        else
            alignment_done(i) <= '0';
            -- TODO: not sure if this works always after a state transition maybe some delay here?
            if ( header_trailer_we = '1' ) then
                -- 37 downto 0  -> hit1
                -- 37 downto 32 -> link number1
                -- 69 downto 38 -> hit2
                -- 75 downto 70 -> link number 2
                -- 77 downto 76 -> header trailer marker
                fifo_data_6(i)(77 downto 76) <= header_trailer(33 downto 32);
                fifo_data_6(i)(75 downto 70) <= (others => '0');
                fifo_data_6(i)(69 downto 38) <= header_trailer(31 downto 0);
                -- padding for now 
                fifo_data_6(i)(37 downto 0) <= tree_paddingk;
                fifo_wen_6(i) <= '1';
            end if;
        end if;
    end if;
    end process;
    END GENERATE tree_layer_last;
    
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
        header_trailer <= (others => '0');
        overflow <= (others => '0');
        rack <= (others => '0');
        header_trailer_we <= '0';
        check_overflow <= '0';
        --
    elsif rising_edge(i_clk) then
        
        rack <= (others => '0');
        header_trailer <= (others => '0');
        header_trailer_we <= '0';
    
        case merge_state is
            when wait_for_pre =>
                -- readout until all fifos have preamble
                if ( sop_wait /= check_zeros ) then
                    wait_cnt_pre <= wait_cnt_pre + '1';
                elsif ( fifo_full_6(0) = '0' ) then
                    merge_state <= get_time1;
                    rack <= (others => '1');
                    -- reset signals
                    wait_cnt_pre <= (others => '0');
                    -- send merged data preamble
                    -- sop & preamble & zeros & datak
                    header_trailer(33 downto 32) <= "01";
                    header_trailer(31 downto 26) <= "111010";
                    header_trailer(7 downto 0) <= x"BC";
                    header_trailer_we <= '1';
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
                if ( error_gtime1 = '0' and fifo_full_6(0) = '0' ) then
                    merge_state <= get_time2;
                    rack <= (others => '1');
                    -- reset signals
                    gtime1 <= (others => (others => '0'));
                    -- send gtime1
                    header_trailer(33 downto 32) <= "00";
                    header_trailer(31 downto 0) <= gtime1(i_link)(35 downto 4);
                    header_trailer_we <= '1';
                elsif ( error_gtime1 = '1' ) then 
                    merge_state <= error_state;
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
                if ( error_gtime2 = '0' and fifo_full_6(0) = '0' ) then
                    merge_state <= wait_for_sh;
                    -- reset signals
                    gtime2 <= (others => (others => '0'));
                    -- send gtime2
                    header_trailer(33 downto 32) <= "00";
                    header_trailer(31 downto 0) <= gtime2(i_link)(35 downto 4);
                    header_trailer_we <= '1';
                elsif ( error_gtime2 = '1' ) then
                    merge_state <= error_state;
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
                elsif ( fifo_full_6(0) = '0' ) then
                    merge_state <= wait_for_sh_written;
                    rack <= (others => '1');
                    -- reset signals
                    wait_cnt_sh <= (others => '0');
                    wait_cnt_merger <= (others => '0');
                    overflow <= (others => '0');
                    -- send merged data sub header
                    -- zeros & sub header & zeros & datak
                    header_trailer(33 downto 32) <= "11";
                    header_trailer(31 downto 28) <= "0000";
                    header_trailer(27 downto 22) <= "111111";
                    -- send sub header time -- check later if equal
                    header_trailer(21 downto 16) <= i_rdata(i_link)(25 downto 20);
                    shtime <= i_rdata(i_link)(25 downto 20);
                    FOR I in N - 1 downto 0 LOOP
                        if ( i_mask_n(I) = '1' ) then
                            sheader_time(I) <= i_rdata(I)(25 downto 20);
                        end if;
                    END LOOP;
                    header_trailer(15 downto 0) <= overflow;
                    header_trailer_we <= '1';
                end if;
                
                -- if wait for pre gets timeout
                if ( wait_cnt_sh = TIMEOUT ) then
                    error_sh <= shop_wait;
                    merge_state <= error_state;
                end if;
                
            when wait_for_sh_written =>
                merge_state <= merge_hits;
                
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
                if ( sh_state = check_zeros and and_reduce(alignment_done) = '1' ) then
                    merge_state <= wait_for_sh;
                end if;
                
                -- TODO error if pre is not there
                if ( pre_state = check_zeros and and_reduce(alignment_done) = '1' ) then
                    merge_state <= wait_for_pre;
                end if;
                
                -- TODO error if trailer is not there
                if ( tr_state = check_zeros and and_reduce(alignment_done) = '1' ) then
                    merge_state <= trailer;
                end if;
                
            when trailer =>
                -- send trailer
                if( fifo_full_6(0) = '0' ) then
                    merge_state <= wait_for_pre;
                    -- send trailer
                    header_trailer(33 downto 32) <= "10";
                    header_trailer(7 downto 0) <= x"9C";
                    header_trailer_we <= '1';
                end if;
                -- reset errors
                error_shtime <= '0';
                error_gtime1 <= '0';
                error_gtime2 <= '0';
                error_merger <= '0';
                error_pre <= (others => '0');
                error_sh <= (others => '0');
                error_tr <= (others => '0');
                                
            when error_state =>
                -- send error message xxxxxxDC
                -- 12: error gtime1
                -- 13: error gtime2
                -- 14: error shtime
                -- N+14 downto 14: error wait for pre
                header_trailer(33 downto 32) <= "11";
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
                header_trailer_we <= '1';
                merge_state <= trailer;
                            
            when others =>
                merge_state <= wait_for_pre;
                
        end case;
        --
    end if;
    end process;

end architecture;
