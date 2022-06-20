-- counters for hits per channel

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


entity ch_rate is
generic (
    num_ch : integer range 0 to 128--;
);
port (

    i_hit       : in  std_logic_vector(35 downto 0);
    i_en        : in  std_logic;

    o_ch_rate   : out work.util.slv32_array_t(num_ch - 1 downto 0);

    i_clk       : in  std_logic;
    i_reset_n   : in  std_logic--;

);
end entity;

architecture rtl of ch_rate is

    signal debug_index : integer range 0 to 128;

begin

    process(i_clk, i_reset_n)
        variable index : integer range 0 to 128;
    begin
    if(i_reset_n /= '1') then
        o_ch_rate <= (others => (others => '0'));
        --
    elsif rising_edge(i_clk) then
        index := to_integer(unsigned(i_hit(29 downto 28) & i_hit(26 downto 22)));
        debug_index <= to_integer(unsigned(i_hit(29 downto 28) & i_hit(26 downto 22)));
        if ( i_hit(33 downto 32) = "00" and i_en = '1' and i_hit(27) = '0' ) then
            o_ch_rate(index) <= o_ch_rate(index) + '1';
        end if;
    end if;
    end process;

end rtl;
