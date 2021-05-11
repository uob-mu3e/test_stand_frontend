library ieee;
use ieee.std_logic_1164.ALL;

entity edge_detector is
Port (
    clk         : in    STD_LOGIC;
    signal_in   : in    STD_LOGIC;
    output      : out   STD_LOGIC
);
end entity;

architecture Behavioral of edge_detector is

    signal signal_d : STD_LOGIC;

begin

    process(clk)
    begin
    if rising_edge(clk) then
        signal_d <= signal_in;
    end if;
    end process;

    output<= (not signal_d) and signal_in;

end architecture;
