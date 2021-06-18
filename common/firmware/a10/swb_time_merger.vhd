library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;


entity swb_time_merger is
generic (
    W : positive := 8*32+8*6;
    TREE_w : positive := 10;
    TREE_r : positive := 10;
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE: std_logic_vector(7 downto 0) := x"01";
    g_NLINKS_DATA : positive := 12;
    g_NLINKS_FARM : positive := 8--;
);
port (
    -- input streams
    i_rx            : in    work.util.slv34_array_t(g_NLINKS_DATA - 1 downto 0);
    i_rsop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_reop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_rshop         : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_rempty        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0) := (others => '1');
    i_rmask_n       : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_rack          : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_counters      : out   work.util.slv32_array_t(0 downto 0);

    -- output strem
    o_q             : out   std_logic_vector(W-1 downto 0);
    o_q_debug       : out   std_logic_vector(31 downto 0);
    o_rempty        : out   std_logic;
    o_rempty_debug  : out   std_logic;
    i_ren           : in    std_logic;
    o_header        : out   std_logic;
    o_header_debug  : out   std_logic;
    o_trailer       : out   std_logic;
    o_trailer_debug : out   std_logic;
    o_error         : out   std_logic;

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of swb_time_merger is

    signal rdata_s, wdata, wdata_reg, fifo_q : std_logic_vector(W-1 downto 0);
    signal fifo_q_debug : std_logic_vector(33 downto 0);
    signal wdata_debug : std_logic_vector(34*g_NLINKS_FARM-1 downto 0);
    signal rdata : work.util.slv38_array_t(g_NLINKS_FARM-1 downto 0);
    signal rempty, wfull, ren, wen, wen_reg : std_logic;
    signal link_number : std_logic_vector(5 downto 0);

    type merge_state_type is (wait_for_pre, get_ts_1, get_ts_2, get_sh, hit, delay, get_tr);
    signal merge_state : merge_state_type;

    signal header_idx  : integer range 0 to 8 := 8;
    signal trailer_idx : integer range 0 to 8 := 8;
    signal ts1_idx     : integer range 0 to 8 := 8;
    signal ts2_idx     : integer range 0 to 8 := 8;
    signal sh_idx      : integer range 0 to 8 := 8;

begin

    --! counters
    e_swb_time_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(0), i_ena => wfull, i_reset_n => i_reset_n, i_clk => i_clk );


    e_time_merger : entity work.time_merger_v3
        generic map (
        W => W,
        TREE_DEPTH_w => TREE_w,
        TREE_DEPTH_r => TREE_r,
        g_NLINKS_DATA => g_NLINKS_DATA,
        DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input streams
        i_rdata                 => i_rx,
        i_rsop                  => i_rsop,
        i_reop                  => i_reop,
        i_rshop                 => i_rshop,
        i_rempty                => i_rempty,
        i_link                  => 1, -- which link should be taken to check ts etc.
        i_mask_n                => i_rmask_n,
        o_rack                  => o_rack,

        -- output stream
        o_rdata                 => rdata_s,
        i_ren                   => ren,--not rempty and not wfull,
        o_empty                 => rempty,
        
        -- error outputs
        o_error_pre             => open,
        o_error_sh              => open,
        o_error_gtime           => open,
        o_error_shtime          => open,
        
        i_reset_n               => i_reset_n,
        i_clk                   => i_clk--,
    );                                                       


    --! map data for time_merger
    generate_rdata : for i in 0 to g_NLINKS_FARM-1 generate
        rdata(i) <= rdata_s(38 * i + 37 downto i * 38);
    end generate;


    --! check for error
    --! TODO: handle errors, at the moment they are sent out at the end of normal events
    o_error     <=  '1' when (rdata(0)(37 downto 32) = err_marker) else
                    '1' when (rdata(2)(37 downto 32) = err_marker) else
                    '1' when (rdata(4)(37 downto 32) = err_marker) else
                    '1' when (rdata(6)(37 downto 32) = err_marker)
                    else '0';

    --! search headers in 256 bit
    header_idx  <=  0 when (rdata(0)(37 downto 32) = pre_marker) else
                    1 when (rdata(1)(37 downto 32) = pre_marker) else
                    2 when (rdata(2)(37 downto 32) = pre_marker) else
                    3 when (rdata(3)(37 downto 32) = pre_marker) else
                    4 when (rdata(4)(37 downto 32) = pre_marker) else
                    5 when (rdata(5)(37 downto 32) = pre_marker) else
                    6 when (rdata(6)(37 downto 32) = pre_marker) else
                    7 when (rdata(7)(37 downto 32) = pre_marker)
                    else 8;
    ts1_idx     <=  0 when (rdata(0)(37 downto 32) = ts1_marker) else
                    1 when (rdata(1)(37 downto 32) = ts1_marker) else
                    2 when (rdata(2)(37 downto 32) = ts1_marker) else
                    3 when (rdata(3)(37 downto 32) = ts1_marker) else
                    4 when (rdata(4)(37 downto 32) = ts1_marker) else
                    5 when (rdata(5)(37 downto 32) = ts1_marker) else
                    6 when (rdata(6)(37 downto 32) = ts1_marker) else
                    7 when (rdata(7)(37 downto 32) = ts1_marker)
                    else 8;
    ts2_idx     <=  0 when (rdata(0)(37 downto 32) = ts2_marker) else
                    1 when (rdata(1)(37 downto 32) = ts2_marker) else
                    2 when (rdata(2)(37 downto 32) = ts2_marker) else
                    3 when (rdata(3)(37 downto 32) = ts2_marker) else
                    4 when (rdata(4)(37 downto 32) = ts2_marker) else
                    5 when (rdata(5)(37 downto 32) = ts2_marker) else
                    6 when (rdata(6)(37 downto 32) = ts2_marker) else
                    7 when (rdata(7)(37 downto 32) = ts2_marker)
                    else 8;
    sh_idx     <=   0 when (rdata(0)(37 downto 32) = sh_marker) else
                    1 when (rdata(1)(37 downto 32) = sh_marker) else
                    2 when (rdata(2)(37 downto 32) = sh_marker) else
                    3 when (rdata(3)(37 downto 32) = sh_marker) else
                    4 when (rdata(4)(37 downto 32) = sh_marker) else
                    5 when (rdata(5)(37 downto 32) = sh_marker) else
                    6 when (rdata(6)(37 downto 32) = sh_marker) else
                    7 when (rdata(7)(37 downto 32) = sh_marker)
                    else 8;
    trailer_idx <=  0 when (rdata(0)(37 downto 32) = tr_marker) else
                    1 when (rdata(1)(37 downto 32) = tr_marker) else
                    2 when (rdata(2)(37 downto 32) = tr_marker) else
                    3 when (rdata(3)(37 downto 32) = tr_marker) else
                    4 when (rdata(4)(37 downto 32) = tr_marker) else
                    5 when (rdata(5)(37 downto 32) = tr_marker) else
                    6 when (rdata(6)(37 downto 32) = tr_marker) else
                    7 when (rdata(7)(37 downto 32) = tr_marker)
                    else 8;

    ren   <= '1' when merge_state = wait_for_pre and header_idx = 8 and rempty = '0' and wfull = '0' else 
             '1' when merge_state = get_ts_1 and ts1_idx = 8 and rempty = '0' and wfull = '0'  else 
             '1' when merge_state = get_ts_2 and ts2_idx = 8 and rempty = '0' and wfull = '0'  else
             '1' when merge_state = get_sh and sh_idx = 8 and rempty = '0' and wfull = '0'  else
             '1' when merge_state = delay and (trailer_idx = 8 or sh_idx = 7) else
             '1' when merge_state = hit and header_idx = 8 and trailer_idx = 8 and ts1_idx = 8 and ts2_idx = 8 and sh_idx = 8 and rempty = '0' and wen_reg = '0' and wfull = '0'  else 
             '0';


    --! read/write data from time merger
    mupix_time_merger : IF DATA_TYPE = x"01" or DATA_TYPE = x"02" GENERATE
        process(i_clk, i_reset_n)
        begin
            if ( i_reset_n = '0' ) then
                --
                wen         <= '0';
                merge_state <= wait_for_pre;
                wdata       <= (others => '0');
                wdata_debug <= (others => '0');
                wdata_reg   <= (others => '1');
                wen_reg     <= '0';
            elsif rising_edge(i_clk) then
                wen         <= '0';
                wen_reg     <= '0';
                wdata_reg   <= (others => '1');
                wdata       <= (others => '0');
                wdata_debug <= (others => '1');
                if ( wfull = '0' and rempty = '0' ) then
                    case merge_state is
                        when wait_for_pre =>
                            if ( header_idx /= 8 ) then
                                merge_state <= get_ts_1;
                                wdata(37 downto 0) <= rdata(header_idx);
                                wdata_debug(31 downto 0) <= rdata(header_idx)(31 downto 0);
                                wdata_debug(33 downto 32) <= "10";
                                wen <= '1';
                            end if;

                        when get_ts_1 =>
                            if ( ts1_idx /= 8 ) then
                                merge_state <= get_ts_2;
                                wdata(37 downto 0) <= rdata(ts1_idx);
                                wdata_debug(31 downto 0) <= rdata(ts1_idx)(31 downto 0);
                                wen <= '1';
                            end if;

                        when get_ts_2 =>
                            if ( ts2_idx /= 8 ) then
                                merge_state <= get_sh;
                                wdata(37 downto 0) <= rdata(ts2_idx);
                                wdata_debug(31 downto 0) <= rdata(ts2_idx)(31 downto 0);
                                wen <= '1';
                            end if;

                        when get_sh =>
                            if ( sh_idx /= 8 ) then
                                merge_state <= delay;
                                wdata(37 downto 0) <= rdata(sh_idx);
                                wdata_debug(31 downto 0) <= rdata(sh_idx)(31 downto 0);
                                wen <= '1';
                            end if;
                            if ( sh_idx /= 7 and sh_idx /= 8 ) then
                                for i in 0 to 7 loop
                                    if ( i >= (sh_idx + 1) and rdata(i) /= tree_paddingk ) then
                                        wdata_reg(38 * i + 37 downto 38 * i) <= rdata(i);
                                        wen_reg <= '1';
                                    end if;
                                end loop;
                            end if;

                        when delay =>
                            if ( wen_reg = '1' ) then
                                wdata <= wdata_reg;
                                for i in 0 to 7 loop
                                    wdata_debug(34 * i + 31 downto 34 * i) <= wdata_reg(38 * i + 31 downto 38 * i);
                                end loop;
                                wen <= '1';
                            end if;
                            merge_state <= hit;

                        when hit =>
                            if ( trailer_idx /= 8 ) then
                                merge_state <= get_tr;
                                for i in 0 to 7 loop
                                    if ( i <= (trailer_idx - 1) ) then
                                        wdata(38 * i + 37 downto 38 * i) <= rdata(i);
                                        wdata_debug(34 * i + 31 downto 34 * i) <= rdata(i)(31 downto 0);
                                    else
                                        wdata(38 * i + 37 downto 38 * i) <= tree_padding;
                                        wdata_debug(34 * i + 31 downto 34 * i) <= tree_padding(31 downto 0);
                                    end if;
                                end loop;
                            elsif ( sh_idx /= 8 ) then
                                merge_state <= get_sh;
                                for i in 0 to 7 loop
                                    if ( i <= (sh_idx - 1) ) then 
                                        wdata(38 * i + 37 downto 38 * i) <= rdata(i);
                                        wdata_debug(34 * i + 31 downto 34 * i) <= rdata(i)(31 downto 0);
                                    else
                                        wdata(38 * i + 37 downto 38 * i) <= tree_padding;
                                        wdata_debug(34 * i + 31 downto 34 * i) <= tree_padding(31 downto 0);
                                    end if;
                                end loop;
                            else
                                for i in 0 to 7 loop
                                    wdata(38 * i + 37 downto 38 * i) <= rdata(i);
                                    wdata_debug(34 * i + 31 downto 34 * i) <= rdata(i)(31 downto 0);
                                end loop;
                            end if;
                            wen <= '1';

                        when get_tr =>
                            if ( trailer_idx /= 8 ) then
                                merge_state <= wait_for_pre;
                                wdata(37 downto 0) <= rdata(trailer_idx);
                                wdata_debug(31 downto 0) <= rdata(trailer_idx)(31 downto 0);
                                wdata_debug(33 downto 32) <= "01";
                                wen <= '1';
                            end if;

                        when others =>
                            merge_state <= wait_for_pre;
                    end case;
                end if;
            end if;
        end process;
    END GENERATE;


    e_swb_time_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => W,
        DEVICE => "Arria 10"--,
    )
    port map (
        q               => fifo_q,
        empty           => o_rempty,
        rdreq           => i_ren,
        data            => wdata,
        full            => wfull,
        wrreq           => wen,--not rempty and not wfull,
        sclr            => not i_reset_n,
        clock           => i_clk--,
    );

    e_debug_fifo : entity work.ip_dcfifo_mixed_widths
    generic map(
        ADDR_WIDTH_w => 8,
        DATA_WIDTH_w => 34*g_NLINKS_FARM,
        ADDR_WIDTH_r => 12,
        DATA_WIDTH_r => 34,
        DEVICE 		 => "Arria 10"--,
    )
    port map (
        aclr 	=> not i_reset_n,
        data 	=> wdata_debug,
        rdclk 	=> i_clk,
        rdreq 	=> i_ren,
        wrclk 	=> i_clk,
        wrreq 	=> wen,
        q 		=> fifo_q_debug,
        rdempty => o_rempty_debug,
        wrfull 	=> open--, -- should be okay since the FIFO above has the same size
    );
    
    o_header    <= '1' when fifo_q(37 downto 32) = pre_marker else '0';
    o_trailer   <= '1' when fifo_q(37 downto 32) = tr_marker else '0';
    o_q         <= fifo_q;

    -- debug path
    o_header_debug  <= '1' when fifo_q_debug(33 downto 32) = "10" else '0';
    o_trailer_debug <= '1' when fifo_q_debug(33 downto 32) = "01" else '0';
    o_q_debug       <= fifo_q_debug(31 downto 0);
    link_number     <= fifo_q(37 downto 32);

end architecture;
