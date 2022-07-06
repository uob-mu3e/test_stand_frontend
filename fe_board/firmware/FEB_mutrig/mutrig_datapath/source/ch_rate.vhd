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

    i_hit       : in  std_logic_vector(33 downto 0);
    i_en        : in  std_logic;

    o_ch_rate   : out work.util.slv32_array_t(num_ch - 1 downto 0);

    i_clk       : in  std_logic;
    i_reset_n   : in  std_logic--;

);
end entity;

architecture rtl of ch_rate is

    signal debug_index  : integer range 0 to num_ch;
    signal ch_counter   : work.util.slv32_array_t(num_ch downto 0);
    signal time_counter : std_logic_vector(31 downto 0);
    signal hit          : std_logic_vector(33 downto 0);
    signal en           : std_logic;

begin

    process(i_clk, i_reset_n)
        variable index : integer range 0 to num_ch;
    begin
    if(i_reset_n /= '1') then
        o_ch_rate       <= (others => (others => '0'));
        ch_counter      <= (others => (others => '0'));
        time_counter    <= (others => '0');
        hit             <= (others => '0');
        en              <= '0';
        --
    elsif rising_edge(i_clk) then
        index := to_integer(unsigned(hit(26 downto 22)));
        hit <= i_hit;
        en  <= i_en;
        -- for simulation only
        debug_index <= to_integer(unsigned(hit(26 downto 22)));
        if ( time_counter = x"7735940" ) then
            for i in 0 to num_ch - 1 loop
                o_ch_rate(i) <= std_logic_vector(resize(unsigned(x"7735940" - ch_counter(i) + 1), 32));
                ch_counter(i) <= (others => '0');
            end loop;
            time_counter <= (others => '0');
        else
            if ( hit(33 downto 32) = "00" and en = '1' and hit(27) = '0' ) then
                ch_counter(index) <= ch_counter(index) + '1';
            end if;
            time_counter <= time_counter + '1';
        end if;
    end if;
    end process;
    
end rtl;
