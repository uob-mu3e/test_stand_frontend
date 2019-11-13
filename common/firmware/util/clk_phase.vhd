--
-- author : Martin Mueller
-- date : 2019.03
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- measure phase between two clocks of _same_ frequency
entity clk_phase is
generic (
    W : positive := 16--;
);
port (
    i_clk1              : in    std_logic;
    i_clk2              : in    std_logic;

    o_phase             : out   std_logic_vector(W-1 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic
);
end entity;

architecture arch of clk_phase is

    signal q : std_logic;

    signal cnt, phase : unsigned(o_phase'range);

begin

    -- sync clock difference to i_clk clock domain
    e_ff_sync : entity work.ff_sync
    generic map ( W => 1 )
    port map (
        d(0) => i_clk2 xor i_clk1,
        q(0) => q,
        rst_n => i_reset_n,
        clk => i_clk--,
    );

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        cnt <= (others => '0');
        phase <= (others => '0');
        o_phase <= (others => '0');
        --
    elsif rising_edge(i_clk) then
        cnt <= cnt + 1;

        if ( q = '1' ) then
            phase <= phase + 1;
        end if;

        if ( cnt = 0 ) then
            phase <= (others => '0');
            o_phase <= std_logic_vector(phase);
        end if;
        --
    end if;
    end process;

end architecture;
