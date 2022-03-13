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


entity link_to_fifo_32 is
generic (
    SKIP_DOUBLE_SUB      : natural := 0;
    LINK_FIFO_ADDR_WIDTH : positive := 10--;
);
port (
    i_rx            : in    work.mu3e.link_t;

    o_q             : out std_logic_vector(33 downto 0);
    i_ren           : in std_logic;
    o_rdempty       : out std_logic;

    --! error counters
    --! 0: fifo almost_full
    --! 1: fifo wrfull
    --! 2: # of skip event
    --! 3: # of events
    --! 4: # of sub header
    o_counter       : out work.util.slv32_array_t(4 downto 0);

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of link_to_fifo_32 is

    type link_to_fifo_type is (idle, write_ts_0, write_ts_1, write_data, skip_data);
    signal link_to_fifo_state : link_to_fifo_type;
    signal cnt_skip_data, cnt_sub, cnt_events : std_logic_vector(31 downto 0);

    signal rx_156_data : std_logic_vector(33 downto 0);
    signal rx_156_wen, almost_full, wrfull : std_logic;
    signal wrusedw : std_logic_vector(LINK_FIFO_ADDR_WIDTH - 1 downto 0);

    signal hit_reg : std_logic_vector(31 downto 0);

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

    --! write only if not idle
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        rx_156_data         <= (others => '0');
        cnt_sub             <= (others => '0');
        cnt_events          <= (others => '0');
        rx_156_wen          <= '0';
        cnt_skip_data       <= (others => '0');
        hit_reg             <= (others => '0');
        link_to_fifo_state  <= idle;
        --
    elsif rising_edge(i_clk) then

        rx_156_wen  <= '0';
        rx_156_data(31 downto 0)  <= i_rx.data;
        rx_156_data(33 downto 32) <= "00";

        if ( i_rx.data = x"000000BC" and i_rx.datak = "0001" ) then
            --
        else
            case link_to_fifo_state is

            when idle =>
                if ( i_rx.data(7 downto 0) = x"BC" and i_rx.datak = "0001" ) then
                    cnt_events <= cnt_events + '1';
                    if ( almost_full = '1' ) then
                        link_to_fifo_state <= skip_data;
                        cnt_skip_data <= cnt_skip_data + '1';
                    else
                        link_to_fifo_state <= write_ts_0;
                        rx_156_data(33 downto 32) <= "10"; -- header
                        rx_156_wen <= '1';
                    end if;
                end if;

            when write_ts_0 =>
                link_to_fifo_state <= write_ts_1;
                rx_156_wen <= '1';

            when write_ts_1 =>
                link_to_fifo_state <= write_data;
                rx_156_wen <= '1';

            when write_data =>
                if ( i_rx.data(7 downto 0) = x"9C" and i_rx.datak = "0001" ) then
                    link_to_fifo_state <= idle;
                    rx_156_data(33 downto 32) <= "01"; -- trailer
                end if;

                if ( i_rx.data(31 downto 26) = "111111" and i_rx.datak = "0000" ) then
                    rx_156_data(33 downto 32) <= "11"; -- sub header
                    cnt_sub <= cnt_sub + '1';
                end if;

                hit_reg <= i_rx.data;

                if ( SKIP_DOUBLE_SUB = 1 and i_rx.data = hit_reg ) then
                    rx_156_wen <= '0';
                else
                    rx_156_wen <= '1';
                end if;

            when skip_data =>
                if ( i_rx.data(7 downto 0) = x"9C" and i_rx.datak = "0001" ) then
                    link_to_fifo_state <= idle;
                end if;

            when others =>
                link_to_fifo_state <= idle;

            end case;
        end if;
        --
    end if;
    end process;

    e_fifo : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => LINK_FIFO_ADDR_WIDTH,
        g_DATA_WIDTH => 34,
        g_WREG_N => 1,
        g_RREG_N => 1--,
    )
    port map (
        i_wdata     => rx_156_data,
        i_we        => rx_156_wen,
        o_wfull     => wrfull,
        o_wusedw    => wrusedw,
        i_wclk      => i_clk,

        o_rdata     => o_q,
        i_rack      => i_ren,
        o_rempty    => o_rdempty,
        i_rclk      => i_clk,

        i_reset_n   => i_reset_n--;
    );

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        almost_full <= '0';
    elsif rising_edge(i_clk) then
        if(wrusedw(LINK_FIFO_ADDR_WIDTH - 1) = '1') then
            almost_full <= '1';
        else
            almost_full <= '0';
        end if;
    end if;
    end process;

end architecture;
