library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


entity time_merger_tree is
generic (
    g_ADDR_WIDTH : positive  := 11;
    N_LINKS_IN   : positive  := 8;
    N_LINKS_OUT  : positive  := 4;
    -- Data type: x"00" = pixel, x"01" = scifi, "10" = tiles
    DATA_TYPE : std_logic_vector(1 downto 0) := "00"--;
);
port (
    -- input data stream
    i_data          : in  work.mu3e.link_array_t(N_LINKS_IN - 1 downto 0);
    i_empty         : in  std_logic_vector(N_LINKS_IN - 1 downto 0);
    i_mask_n        : in  std_logic_vector(N_LINKS_IN - 1 downto 0);
    o_rack          : out std_logic_vector(N_LINKS_IN - 1 downto 0);

    -- output data stream
    o_data          : out work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0);
    o_empty         : out std_logic_vector(N_LINKS_OUT - 1 downto 0);
    o_mask_n        : out std_logic_vector(N_LINKS_OUT - 1 downto 0);
    i_rack          : in  std_logic_vector(N_LINKS_OUT - 1 downto 0);

    -- counters
    -- 0: HEADER counters
    -- 1: SHEADER counters
    -- 2: HIT counters
    o_counters      : out work.util.slv32_array_t(3 * N_LINKS_OUT - 1 downto 0);

    i_en            : in  std_logic;
    i_reset_n       : in  std_logic;
    i_clk           : in  std_logic--;
);
end entity;

architecture arch of time_merger_tree is

    signal reset_n, en_reg : std_logic;

    -- merger signals
    constant size : integer := N_LINKS_IN/2;
    signal mask_n : std_logic_vector(N_LINKS_IN - 1 downto 0);

    -- layer states
    signal layer_state, last_state : work.util.slv4_array_t(N_LINKS_OUT - 1 downto 0);
    signal state_reset : std_logic;

    -- fifo signals
    signal data, q_data : work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0);
    signal wrreq, wrfull : std_logic_vector(N_LINKS_OUT - 1 downto 0);

    -- hit signals
    signal a, b : work.util.slv4_array_t(N_LINKS_OUT - 1 downto 0) := (others => (others => '1'));
    signal a_h, b_h : work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0);
    signal overflow : work.util.slv16_array_t(N_LINKS_OUT - 1 downto 0) := (others => (others => '0'));

    -- error signals
    signal shop_time0, shop_time1 : work.util.slv7_array_t(N_LINKS_OUT - 1 downto 0) := (others => (others => '0'));
    signal error_s : work.util.slv2_array_t(N_LINKS_OUT - 1 downto 0) := (others => (others => '0'));

    -- counters
    signal countSop, countSbhdr, countHit : std_logic_vector(N_LINKS_OUT - 1 downto 0);

    -- default link types
    signal headerHit : work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0) := (others => work.mu3e.LINK_SOP);
    signal sbhdrHit : work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0) := (others => work.mu3e.LINK_SBHDR);
    signal trailerHit : work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0) := (others => work.mu3e.LINK_EOP);
    signal errorHit : work.mu3e.link_array_t(N_LINKS_OUT - 1 downto 0) := (others => work.mu3e.LINK_ERR);

begin

    e_reset_n : entity work.reset_sync
    port map ( o_reset_n => reset_n, i_reset_n => i_reset_n, i_clk => i_clk );

    --! reset / enable for tree
    process(i_clk, reset_n)
    begin
    if ( reset_n /= '1' ) then
        state_reset <= '1';
        en_reg <= '0';
        --
    elsif ( rising_edge(i_clk) ) then
        state_reset <= '0';
        en_reg <= i_en;
    end if;
    end process;

    gen_hits:
    FOR i in 0 to N_LINKS_OUT - 1 GENERATE

        --! HEADER counters
        countSop(i) <= '1' when layer_state(i) = HEADER else '0';
        e_cnt_header_state : entity work.counter
        generic map (
            WRAP => true,
            W => o_counters(0+i*3)'length--,
        )
        port map (
            o_cnt => o_counters(0+i*3),
            i_ena => countSop(i),
            i_reset_n => reset_n,
            i_clk => i_clk--,
        );

        --! SHEADER counters
        countSbhdr(i) <= '1' when layer_state(i) = SHEADER else '0';
        e_cnt_sheader_state : entity work.counter
        generic map (
            WRAP => true,
            W => o_counters(1+i*3)'length--,
        )
        port map (
            o_cnt => o_counters(1+i*3),
            i_ena => countSbhdr(i),
            i_reset_n => reset_n,
            i_clk => i_clk--,
        );

        --! HIT counters
        countHit(i) <= '1' when layer_state(i) = HIT else '0';
        e_cnt_hit_state : entity work.counter
        generic map (
            WRAP => true,
            W => o_counters(2+i*3)'length--,
        )
        port map (
            o_cnt => o_counters(2+i*3),
            i_ena => countHit(i),
            i_reset_n => reset_n,
            i_clk => i_clk--,
        );

        mupix_data : IF DATA_TYPE = "00" GENERATE
            a(i)    <= i_data(i).data(31 downto 28) when mask_n(i) = '1' else (others => '1');
            b(i)    <= i_data(i+size).data(31 downto 28) when mask_n(i+size) = '1' else (others => '1');
        END GENERATE;
        
        scifi_data : IF DATA_TYPE = "01" GENERATE
            a(i)    <= i_data(i).data(9 downto 6) when mask_n(i) = '1' else (others => '1');
            b(i)    <= i_data(i+size).data(9 downto 6) when mask_n(i+size) = '1' else (others => '1');
        END GENERATE;

        a_h(i)      <= i_data(i);
        b_h(i)      <= i_data(i+size);

        --! reg mask for timing
        process(reset_n, i_clk)
        begin
        if ( reset_n /= '1' ) then
            mask_n(i) <= '0';
            o_mask_n(i) <= '0';
            --
        elsif ( rising_edge(i_clk) ) then
            mask_n(i)           <= i_mask_n(i);
            mask_n(i + size)    <= i_mask_n(i + size);
            o_mask_n(i)         <= i_mask_n(i) or i_mask_n(i + size);
        end if;
        end process;

        e_tree_fifo : entity work.link_scfifo
        generic map (
            g_ADDR_WIDTH => g_ADDR_WIDTH,
            g_WREG_N => 2,
            g_RREG_N => 2--,
        )
        port map (
            i_wdata         => data(i),
            i_we            => wrreq(i),
            o_wfull         => wrfull(i),

            o_rdata         => q_data(i),
            o_rempty        => o_empty(i),
            i_rack          => i_rack(i),

            i_clk           => i_clk,
            i_reset_n       => reset_n--,
        );

        o_data(i)  <= q_data(i) when o_empty(i) = '0' else work.mu3e.LINK_ZERO;

        -- Tree setup
        -- x => empty, h => header, t => time header, tr => trailer, sh => sub header
        -- [a]               [a]                   [a]
        -- [1]  -> [[2],[1]] [tr]  -> [[tr],[2]]   [4,sh]   -> [[4],[3],[sh],[2]]
        -- [2]               [tr,2]                [3,sh,2]
        -- [b]               [b]                   [b]
        layer_state(i) <=             -- check if both are mask or if we are in enabled or in reset
                            SWB_IDLE  when (mask_n(i) = '0' and mask_n(i+size) = '0') or en_reg = '0' or state_reset = '1' else
                                      -- we forword the error the chain
                            ONEERROR  when (i_data(i).err = '1' or i_data(i+size).err = '1') and wrfull(i) = '0' else
                                      -- simple case on of the links is mask so we just send the other throw the tree
                            ONEMASK   when (mask_n(i) = '0' or mask_n(i+size) = '0') and wrfull(i) = '0' else
                                      -- wait if one input is empty or the output fifo is full
                            WAITING   when i_empty(i) = '1' or i_empty(i+size) = '1' or wrfull(i) = '1' else
                                      -- since we check in before that we should have two links not masked and both are not empty we 
                                      -- want to see from both a header
                            HEADER    when i_data(i).sop = '1' and i_data(i+size).sop = '1' and (last_state(i) = SWB_IDLE or last_state(i) = TRAILER) else
                                      -- we now want that both hits have ts0
                            TS0       when i_data(i).t0 = '1' and i_data(i+size).t0 = '1' and last_state(i) = HEADER else
                                      -- we now want that both hits have ts1
                            TS1       when i_data(i).t1 = '1' and i_data(i+size).t1 = '1' and last_state(i) = TS0 else
                                      -- we check if both hits have a subheader
                            SHEADER   when i_data(i).sbhdr = '1' and i_data(i+size).sbhdr = '1' and (last_state(i) = TS1 or last_state(i) = HIT or last_state(i) = ONEHIT or last_state(i) = SHEADER) else
                                      -- we check if both hits have a hit
                            HIT       when i_data(i).dthdr = '1' and i_data(i+size).dthdr = '1' and (last_state(i) = SHEADER or last_state(i) = HIT) else
                                      -- we check if one has a subheader or trailer and the other link has a hit
                            ONEHIT    when ((i_data(i).dthdr = '1' and (i_data(i+size).sbhdr = '1' or i_data(i+size).eop = '1')) or ((i_data(i+size).dthdr = '1' and (i_data(i).sbhdr = '1' or i_data(i).eop = '1')))) and (last_state(i) = SHEADER or last_state(i) = HIT or last_state(i) = ONEHIT) else
                                      -- we check if both hits have a trailer
                            TRAILER   when i_data(i).eop = '1' and i_data(i+size).eop = '1' and (last_state(i) = SHEADER or last_state(i) = HIT or last_state(i) = ONEHIT) else
                            WAITING;
                            
        -- TODO: simplifiy same when cases
        -- NOTE: if timing problem maybe add reg for writing

        wrreq(i)        <=  '1' when layer_state(i) = HEADER or layer_state(i) = TS0 or layer_state(i) = TS1 or layer_state(i) = SHEADER or layer_state(i) = HIT or layer_state(i) = ONEHIT or layer_state(i) = TRAILER or layer_state(i) = ONEERROR else
                            not i_empty(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' else
                            not i_empty(i+size) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' else
                            '0';

        o_rack(i)       <=  '1' when layer_state(i) = HEADER or layer_state(i) = TS0 or layer_state(i) = TS1 or layer_state(i) = SHEADER or layer_state(i) = TRAILER else
                            '1' when layer_state(i) = ONEHIT and i_data(i).dthdr = '1' else
                            '1' when layer_state(i) = HIT and a(i) <= b(i) else
                            not i_empty(i) when layer_state(i) = ONEERROR and i_data(i).err = '1' else
                            not i_empty(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' else
                            '0';

        o_rack(i+size)  <=  '1' when layer_state(i) = HEADER or layer_state(i) = TS0 or layer_state(i) = TS1 or layer_state(i) = SHEADER or layer_state(i) = TRAILER else
                            '1' when layer_state(i) = ONEHIT and i_data(i+size).dthdr = '1' else
                            '1' when layer_state(i) = HIT and b(i) < a(i) else
                            not i_empty(i+size) when layer_state(i) = ONEERROR and i_data(i+size).err = '1' else
                            not i_empty(i+size) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' else
                            '0';

        -- or'ed overflow
        overflow(i) <=  a_h(i).data(15 downto 0)  or b_h(i).data(15 downto 0)  when layer_state(i) = SHEADER else
                        a_h(i).data(31 downto 16) or b_h(i).data(31 downto 16) when layer_state(i) = TRAILER else
                        (others => '0');

        -- do some error checking
        shop_time0(i) <= a_h(i).data(29 downto 28) & a_h(i).data(20 downto 16);
        shop_time1(i) <= b_h(i).data(29 downto 28) & b_h(i).data(20 downto 16);
        error_s(i)    <= "01"   when layer_state(i) = TS0 and a_h(i).data /= b_h(i).data else
                         "10"   when layer_state(i) = TS1 and a_h(i).data(31 downto 27) /= b_h(i).data(31 downto 27) else
                         "11"   when layer_state(i) = SHEADER and shop_time0(i) /= shop_time1(i) else
                         (others => '0');
                         
        errorHit(i).err <=  '1' when layer_state(i) = ONEERROR and i_data(i).err = '1' else
                            '1' when layer_state(i) = ONEERROR and i_data(i+size).err = '1' else
                            '1' when work.util.or_reduce(error_s(i)) = '1' else
                            '0';

        -- set data for no hit types
        headerHit(i).data  <= x"E80000BC";
        sbhdrHit(i).data   <= (a_h(i).data(31 downto 16) or b_h(i).data(31 downto 16)) & overflow(i);
        trailerHit(i).data <= overflow(i) & x"009C";
        errorHit(i).data   <= error_s(i) & "00" & x"FFFFF9C";

        -- synthesis translate_off
        assert ( work.util.or_reduce(error_s(i)) = '1'
        ) report "Tree ERROR"
            & ", a_h = " & work.util.to_hstring(a_h(i).data)
            & ", b_h = " & work.util.to_hstring(b_h(i).data)
            & ", a_ts = " & work.util.to_hstring(shop_time0(i))
            & ", b_ts = " & work.util.to_hstring(shop_time1(i))
            & ", sbhdr = " & work.util.to_hstring(a_h(i).sbhdr & b_h(i).sbhdr)
            & ", layer_state = " & work.util.to_hstring(layer_state(i))
            & ", last_state = " & work.util.to_hstring(last_state(i))
            & ", error = " & work.util.to_hstring(error_s(i))
        severity note;
        -- synthesis translate_on

        -- write out data
        data(i)         <=  errorHit(i) when work.util.or_reduce(error_s(i)) = '1' else
                            headerHit(i) when layer_state(i) = HEADER else
                            a_h(i) when layer_state(i) = TS0 else
                            a_h(i) when layer_state(i) = TS1 else
                            sbhdrHit(i) when layer_state(i) = SHEADER else
                            trailerHit(i) when layer_state(i) = TRAILER else
                            a_h(i) when layer_state(i) = ONEHIT and i_data(i).dthdr = '1' else
                            b_h(i) when layer_state(i) = ONEHIT and i_data(i+size).dthdr = '1' else
                            a_h(i) when layer_state(i) = HIT and a(i) <= b(i) else
                            b_h(i) when layer_state(i) = HIT and b(i) < a(i) else
                            a_h(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' and i_data(i).sop = '1' else
                            a_h(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' and i_data(i).t0   = '1' else
                            a_h(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' and i_data(i).t1   = '1' else
                            a_h(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' and i_data(i).sbhdr = '1' else
                            a_h(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' and i_data(i).dthdr  = '1' else
                            a_h(i) when layer_state(i) = ONEMASK and mask_n(i) = '1' and i_data(i).eop = '1' else
                            b_h(i) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' and i_data(i+size).sop = '1' else
                            b_h(i) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' and i_data(i+size).t0   = '1' else
                            b_h(i) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' and i_data(i+size).t1   = '1' else
                            b_h(i) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' and i_data(i+size).sbhdr = '1' else
                            b_h(i) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' and i_data(i+size).dthdr  = '1' else
                            b_h(i) when layer_state(i) = ONEMASK and mask_n(i+size) = '1' and i_data(i+size).eop = '1' else
                            work.mu3e.LINK_ZERO;

        -- set last layer state
        process(i_clk, reset_n)
        begin
        if ( reset_n /= '1' ) then
            last_state(i)   <= SWB_IDLE;
            --
        elsif ( rising_edge(i_clk) ) then
            -- TODO: should we do a counter here -> leading to an error?
            if ( layer_state(i) /= WAITING ) then
                last_state(i) <= layer_state(i);
            end if;
        end if;
        end process;

    END GENERATE;

end architecture;
