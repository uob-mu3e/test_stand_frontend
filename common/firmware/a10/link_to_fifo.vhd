-------------------------------------------------------
--! @link_to_fifo_32.vhd
--! @brief the link_to_fifo_32 sorts out the data from the
--! link and provides it as a fifo (32b package)
--! Author: mkoeppel@uni-mainz.de
-------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity link_to_fifo is
generic (
    g_LOOPUP_NAME        : string := "intRun2021";
    is_FARM              : boolean := false;
    SKIP_DOUBLE_SUB      : boolean := false;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in  work.mu3e.link_t;
    i_linkid        : in  std_logic_vector(5 downto 0);  
    
    o_q             : out work.mu3e.link_t;
    i_ren           : in  std_logic;
    o_rdempty       : out std_logic;

    --! error counters
    --! 0: fifo almost_full
    --! 1: fifo wrfull
    --! 2: # of skip event
    --! 3: # of events
    --! 4: # of sub header
    o_counter       : out work.util.slv32_array_t(4 downto 0);

    i_reset_n       : in  std_logic;
    i_clk           : in  std_logic--;
);
end entity;

architecture arch of link_to_fifo is

    type link_to_fifo_type is (idle, write_ts_0, write_ts_1, write_data, skip_data);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_data, cnt_sub, cnt_events : std_logic_vector(31 downto 0);

    signal rx : work.mu3e.link_t;
    signal rx_wen, almost_full, wrfull : std_logic;
    signal wrusedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH - 1 downto 0);

    signal hit_reg : std_logic_vector(31 downto 0);
    signal chipID : std_logic_vector(6 downto 0);

begin

    e_cnt_link_fifo_almost_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(0), i_ena => almost_full, i_reset_n => i_reset_n, i_clk => i_clk );

    e_cnt_dc_link_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counter(1), i_ena => wrfull, i_reset_n => i_reset_n, i_clk => i_clk );

    o_counter(2) <= cnt_skip_data;
    o_counter(3) <= cnt_events;
    o_counter(4) <= cnt_sub;
    
    -- replace chip id
    e_lookup : entity work.chip_lookup
    generic map ( g_LOOPUP_NAME => g_LOOPUP_NAME )
    port map ( i_fpgaID => i_linkid, i_chipID => i_rx.data(25 downto 22), o_chipID => chipID );

    --! write only if not idle
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        rx                  <= work.mu3e.LINK_ZERO;
        cnt_sub             <= (others => '0');
        cnt_events          <= (others => '0');
        rx_wen              <= '0';
        cnt_skip_data       <= (others => '0');
        hit_reg             <= (others => '0');
        link_to_fifo_state  <= idle;
        --
    elsif ( rising_edge(i_clk) ) then

        rx_wen <= '0';
        rx <= i_rx;
        -- reset sop/eop/sh
        rx.sop <= '0';
        rx.eop <= '0';
        rx.sbhdr <= '0';
        rx.t0 <= '0';
        rx.t1 <= '0';
        rx.dthdr <= '0';
        rx.err <= '0';
        
        hit_reg <= i_rx.data;

        if ( i_rx.idle = '1' ) then
            --
        else
            case link_to_fifo_state is

            when idle =>
                if ( i_rx.sop = '1' ) then
                    cnt_events <= cnt_events + '1';
                    if ( almost_full = '1' ) then
                        link_to_fifo_state <= skip_data;
                        cnt_skip_data <= cnt_skip_data + '1';
                    else
                        link_to_fifo_state <= write_ts_0;
                        -- header
                        rx.sop <= '1';
                        rx_wen <= '1';
                    end if;
                end if;

            when write_ts_0 =>
                link_to_fifo_state <= write_ts_1;
                rx.t0 <= '1';
                rx_wen <= '1';

            when write_ts_1 =>
                link_to_fifo_state <= write_data;
                rx.t1 <= '1';
                rx_wen <= '1';

            when write_data =>
                if ( i_rx.eop = '1' ) then
                    link_to_fifo_state <= idle;
                    rx.eop <= '1';
                -- check for sub header on the SWB
                elsif ( i_rx.data(31 downto 26) = "111111" and i_rx.datak = "0000" and not is_FARM ) then
                    -- we shift the subheader around here the marker will be 1111111 for chipID = 128
                    -- on position 27 downto 21, overflow will be 15 downto 0 and the time stamp
                    -- will be shifted from ts(10-9) to 29-28 and from ts(8-4)to 20-16
                    rx.data <= "00" & i_rx.data(22 downto 21) & "1111111" & i_rx.data(20 downto 16) & i_rx.data(15 downto 0); -- sub header
                    rx.sbhdr <= '1';
                    cnt_sub <= cnt_sub + '1';
                -- write hit on swb
                elsif ( not is_FARM ) then
                    rx.data <= i_rx.data(31 downto 28) & chipID & i_rx.data(21 downto 1); -- hit
                    rx.dthdr <= '1';
                -- check for sub header on the farm
                elsif ( i_rx.data(27 downto 21) = "1111111" and i_rx.datak = "0000" and is_FARM ) then
                    rx.sbhdr <= '1';
                    cnt_sub <= cnt_sub + '1';
                -- write hit on farm
                elsif ( is_FARM ) then
                    -- TODO: changing to 64bit hit later
                    rx.dthdr <= '1';
                end if;
                
                if ( SKIP_DOUBLE_SUB and i_rx.data = hit_reg ) then
                    rx_wen <= '0';
                else
                    rx_wen <= '1';
                end if;

                -- TODO: throw away subheader hits if the tree is not merging
            when skip_data =>
                if ( i_rx.eop = '1' ) then
                    link_to_fifo_state <= idle;
                end if;

            when others =>
                link_to_fifo_state <= idle;

            end case;
        end if;
        --
    end if;
    end process;

    e_fifo : entity work.link_scfifo
    generic map (
        g_ADDR_WIDTH=> LINK_FIFO_ADDR_WIDTH,
        g_WREG_N    => 1,
        g_RREG_N    => 1--,
    )
    port map (
        i_wdata     => rx,
        i_we        => rx_wen,
        o_wfull     => wrfull,
        o_usedw     => wrusedw,

        o_rdata     => o_q,
        i_rack      => i_ren,
        o_rempty    => o_rdempty,
        
        i_clk       => i_clk,
        i_reset_n   => i_reset_n--;
    );

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        almost_full <= '0';
    elsif ( rising_edge(i_clk) ) then
        if ( wrusedw(LINK_FIFO_ADDR_WIDTH - 1) = '1' ) then
            almost_full <= '1';
        else
            almost_full <= '0';
        end if;
    end if;
    end process;

end architecture;
