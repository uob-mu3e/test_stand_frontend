library ieee;
use ieee.std_logic_1164.all;

entity rx_reset is
    generic (
        Nch : positive := 4;
        CLK_MHZ : positive := 50--;
    );
    port (
        -- reset the receiver CDR present in the receiver channel
        analogreset     :   out std_logic_vector(Nch-1 downto 0);
        -- reset all digital logic in the receiver PCS
        digitalreset    :   out std_logic_vector(Nch-1 downto 0);

        ready           :   out std_logic_vector(Nch-1 downto 0);

        -- status of the receiver CDR lock mode
        freqlocked      :   in  std_logic_vector(Nch-1 downto 0);

        -- status of the dynamic reconfiguration controller
        reconfig_busy   :   in  std_logic;

        arst_n  :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity;

architecture arch of rx_reset is

    constant LTD_WIDTH : positive := 4000; -- ns

    signal rst_n : std_logic;

    signal analogreset_n : std_logic;
    signal digitalreset_n : std_logic_vector(Nch-1 downto 0);

begin

    i_rst_n : entity work.reset_sync
    port map ( rstout_n => rst_n, arst_n => arst_n, clk => clk );

    analogreset <= (others => not analogreset_n);
    digitalreset <= not digitalreset_n;

    i_analogreset_n : entity work.reset_sync
    port map (
        rstout_n => analogreset_n,
        arst_n => rst_n and not reconfig_busy,
        clk => clk--,
    );

    g_digitalreset_n : for i in digitalreset_n'range generate
    begin
    i_digitalreset_n : entity work.debouncer
    generic map ( W => 1, N => LTD_WIDTH * CLK_MHZ / 1000 )
    port map (
        d(0) => '1', q(0) => digitalreset_n(i),
        arst_n => rst_n and freqlocked(i),
        clk => clk--,
    );
    end generate;

    ready <= digitalreset_n;

end architecture;
