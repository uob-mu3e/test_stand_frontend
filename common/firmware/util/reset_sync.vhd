--
-- author : Alexandr Kozlinskiy
--

library ieee;
use ieee.std_logic_1164.all;

-- reset synchronizer
entity reset_sync is
    generic (
        -- number of stages
        N : positive := 2--;
    );
    port (
        rstout_n    :   out std_logic;
        arst_n      :   in  std_logic;
        clk         :   in  std_logic--;
    );
end entity;

architecture arch of reset_sync is

    -- no 'preserve' attr on final output reg
    -- (see altera_reset_synchronizer)
    signal q : std_logic;

begin

    i_ff_sync : entity work.ff_sync
    generic map ( W => 1, N => N )
    port map ( i_d(0) => '1', o_q(0) => q, i_reset_n => arst_n, i_clk => clk );

    process(clk, arst_n)
    begin
    if ( arst_n = '0' ) then
        rstout_n <= '0';
    elsif rising_edge(clk) then
        rstout_n <= q;
    end if;
    end process;

end architecture;
