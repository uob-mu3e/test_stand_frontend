--
-- author : Marius Koeppel
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- clock divider
-- period_{o_clk} = P * period_{i_clk}
entity clkdiv_dynamic is
port (
    o_clk       : out   std_logic;
    i_reset_n   : in    std_logic;
    i_P         : in    std_logic_vector(31 downto 0);
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of clkdiv_dynamic is

    signal clk1_even, clk1_odd, clk2_odd, clk_even, clk_odd : std_logic;
    signal P          : positive := 1;
    signal cnt1_even, cnt1_odd, cnt2_odd : integer;

begin

    P <= to_integer(unsigned(i_P));
    --! output clk
    o_clk <=    i_clk       when P = 1 else
                clk_even    when P > 1 and P mod 2 = 0 else
                clk_odd     when P > 1 and P mod 2 = 1 else
                i_clk;


    --! even process
    clk_even <= clk1_even;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        clk1_even <= '0';
        cnt1_even <= 0;
        --
    elsif rising_edge(i_clk) then
        if ( cnt1_even = P/2-1 or cnt1_even = P-1 ) then
            cnt1_even <= 0;
            clk1_even <= not clk1_even;
        else
            cnt1_even <= cnt1_even + 1;
        end if;
        --
    end if;
    end process;


    --! odd process
    clk_odd <= clk1_odd and clk2_odd;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        clk1_odd <= '0';
        cnt1_odd <= 0;
        --
    elsif rising_edge(i_clk) then
        if ( cnt1_odd = 0 or cnt1_odd = (P+1)/2 ) then
            clk1_odd <= not clk1_odd;
        end if;
        if ( cnt1_odd = P-1 ) then
            cnt1_odd <= 0;
        else
            cnt1_odd <= cnt1_odd + 1;
        end if;
        --
    end if;
    end process;

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        clk2_odd <= '0';
        cnt2_odd <= 0;
        --
    elsif falling_edge(i_clk) then
        if ( cnt2_odd = 0 or cnt2_odd = (P+1)/2 ) then
            clk2_odd <= not clk2_odd;
        end if;
        if ( cnt2_odd = P-1 ) then
            cnt2_odd <= 0;
        else
            cnt2_odd <= cnt2_odd + 1;
        end if;
        --
    end if;
    end process;

end architecture;
