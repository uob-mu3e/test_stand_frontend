library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_uart is
end entity;

architecture arch of tb_uart is

    constant CLK_MHZ : real := 100.0; -- MHz
    signal clk, reset_n : std_logic := '0';

    type data_t is array (natural range <>) of std_logic_vector(7 downto 0);
    signal data : data_t(3 downto 0) := (
        X"01",
        others => (others => '0')
    );

    signal data_tx_to_rx : std_logic;
    signal we, wfull : std_logic;
    signal rempty, rack : std_logic;

    signal DONE : std_logic_vector(2 downto 0) := (others => '0');

begin

    clk <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    e_uart_tx : entity work.uart_tx
    generic map (
        STOP_BITS_g => 2,
        BAUD_RATE_g => 10000000,
        CLK_MHZ_g => CLK_MHZ--,
    )
    port map (
        o_data          => data_tx_to_rx,

        i_wdata         => X"AA",
        i_we            => we,
        o_wfull         => wfull,

        i_reset_n       => reset_n,
        i_clk           => clk--,
    );
    we <= not wfull;

    e_uart_rx : entity work.uart_rx
    generic map (
        STOP_BITS_g => 2,
        BAUD_RATE_g => 10000000,
        CLK_MHZ_g => CLK_MHZ--,
    )
    port map (
        i_data          => data_tx_to_rx,

        o_rempty        => rempty,
        i_rack          => rack,

        i_reset_n       => reset_n,
        i_clk           => clk--,
    );
    rack <= not rempty;

    process
    begin
        wait;
    end process;

    process
    begin
        wait for 1000 ns;
        assert ( DONE = (DONE'range => '1') ) severity error;
        wait;
    end process;

end architecture;
