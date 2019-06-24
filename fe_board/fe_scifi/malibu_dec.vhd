library ieee;
use ieee.std_logic_1164.all;

entity malibu_dec is
generic (
    N : positive := 2--,
);
port (
    data_clk    : out   std_logic;
    data        : out   std_logic_vector(8*N-1 downto 0);
    datak       : out   std_logic_vector(N-1 downto 0);

    -- serial data (N channels)
    rx_data : in    std_logic_vector(N-1 downto 0);
    rx_clk  : in    std_logic;

    reset   : in    std_logic--;
);
end entity;

architecture arch of malibu_dec is

    constant K28_5 : std_logic_vector(9 downto 0) := "0101" & "111100";

    signal rx_out : std_logic_vector(10*N-1 downto 0); -- 10 bits * N channels
    signal rx_channel_data_align_n : std_logic_vector(N-1 downto 0);

    signal data_clk_i : std_logic;
    type data10_t is array ( natural range <> ) of std_logic_vector(9 downto 0);
    signal data10 : data10_t (0 to N-1);

begin

    data_clk <= data_clk_i;

    i_lvds_rx : entity work.ip_altlvds_rx
    generic map (
        N => N,
        W => 10,
        PLL_FREQ => 160,
        DATA_RATE => 160--,
    )
    port map (
        rx_channel_data_align => not rx_channel_data_align_n(1 downto 0),
        rx_in => rx_data,
        rx_inclock => rx_clk,
        rx_reset => (others => reset),
        rx_dpa_locked => open,
        rx_locked => open,
        rx_out => rx_out,
        rx_outclock => data_clk_i--,
    );

    g_rx : for i in rx_data'range generate
    begin
        data10(i) <= work.util.reverse(rx_out(9 + 10*i downto 0 + 10*i));

        -- pulse data align if no K
        i_isK : entity work.watchdog
        port map (
            d(0) => work.util.to_std_logic(data10(i) = K28_5 or data10(i) = not K28_5),
            rstout_n => rx_channel_data_align_n(i),
            rst_n => not reset,
            clk => data_clk_i--,
        );

        i_dec_8b10b : entity work.dec_8b10b
        port map (
            AI => rx_out(9+10*i), BI => rx_out(8+10*i), CI => rx_out(7+10*i), DI => rx_out(6+10*i), EI => rx_out(5+10*i), II => rx_out(4+10*i),
            FI => rx_out(3+10*i), GI => rx_out(2+10*i), HI => rx_out(1+10*i), JI => rx_out(0+10*i),
            KO => datak(i),
            HO => data(7+8*i), GO => data(6+8*i), FO => data(5+8*i), EO => data(4+8*i), DO => data(3+8*i), CO => data(2+8*i), BO => data(1+8*i), AO => data(0+8*i),
            RESET => not rx_channel_data_align_n(0),
            RBYTECLK => data_clk_i--,
        );
    end generate;

end architecture;
