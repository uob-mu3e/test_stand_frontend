library ieee;
use ieee.std_logic_1164.all;

entity tx_reset is
    generic (
        Nch : positive := 4;
        Npll : positive := 1;
        CLK_MHZ : positive := 50--;
    );
    port (
        analogreset     :   out std_logic_vector(Nch-1 downto 0);
        -- asynchronous reset to all digital logic in the transmitter PCS
        digitalreset    :   out std_logic_vector(Nch-1 downto 0);

        ready           :   out std_logic_vector(Nch-1 downto 0);

        -- powers down the CMU PLLs
        pll_powerdown   :   out std_logic_vector(Npll-1 downto 0);
        -- status of the transmitter PLL
        pll_locked      :   in  std_logic_vector(Npll-1 downto 0);

        arst_n  :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity;

architecture arch of tx_reset is

    constant PLL_POWERDOWN_WIDTH : positive := 1000; -- ns

    signal pll_powerdown_n : std_logic;
    signal analogreset_n : std_logic;
    signal digitalreset_n : std_logic;

begin

    pll_powerdown <= (others => not pll_powerdown_n);
    analogreset <= (others => not analogreset_n);
    digitalreset <= (others => not digitalreset_n);

    i_pll_powerdown_n : entity work.debouncer
    generic map ( W => 1, N => PLL_POWERDOWN_WIDTH * CLK_MHZ / 1000 )
    port map (
        d(0) => '1', q(0) => pll_powerdown_n,
        arst_n => arst_n,
        clk => clk--,
    );

    analogreset_n <= pll_powerdown_n;

    i_digitalreset_n : entity work.reset_sync
    port map (
        rstout_n => digitalreset_n,
        arst_n => pll_powerdown_n and work.util.and_reduce(pll_locked),
        clk => clk--,
    );

    ready <= (others => digitalreset_n);

end architecture;
