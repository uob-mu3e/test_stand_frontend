--
-- author : Alexandr Kozlinskiy
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity counter is
    generic (
        W : positive := 8;
        DIV : positive := 1;
        EDGE : integer := 0;
        DIR : integer := 1;
        WRAP : boolean := false--;
    );
    port (
        cnt     :   out std_logic_vector(W-1 downto 0);
        ena     :   in  std_logic;

        reset   :   in  std_logic;
        clk     :   in  std_logic--;
    );
end entity;

architecture arch of counter is

    signal ena_q, ena_i : std_logic;

    signal cnt0 : integer range 0 to DIV-1;
    signal cnt1 : integer range 0 to 2**W-1;

begin

    ena_i <= ena and not ena_q when EDGE > 0 else
             not ena and ena_q when EDGE < 0 else
             ena;

    cnt <= std_logic_vector(to_unsigned(cnt1, W));

    process(clk)
    begin
    if rising_edge(clk) then
        ena_q <= ena;

        if ( reset = '1' ) then
            cnt0 <= 0;
            cnt1 <= 0;
        elsif ( ena_i = '1' ) then
            if ( DIV > 1 and cnt0 /= DIV-1 ) then
                cnt0 <= cnt0 + 1;
            else
                cnt0 <= 0;
                if ( cnt1 /= 2**W-1 ) then
                    cnt1 <= cnt1 + 1;
                elsif ( WRAP ) then
                    cnt1 <= 0;
                end if;
            end if;
        end if;
        --
    end if;
    end process;

end architecture;
