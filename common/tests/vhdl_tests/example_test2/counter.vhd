
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity counter is
generic (
    W : positive := 8--;
);
port (
    o_cnt       : out   std_logic_vector(W-1 downto 0);
    i_ena       : in    std_logic;

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of counter is

    signal cnt : unsigned(W-1 downto 0);

begin

    o_cnt <= std_logic_vector(cnt);

    process(i_clk, i_reset_n, i_ena)
    begin
    if ( i_reset_n = '0' ) then
        cnt <= (others => '0');
        --
    elsif ( rising_edge(i_clk) and i_ena = '1' ) then
        cnt <= cnt + 1;
        --
    end if;
    end process;

end architecture;
