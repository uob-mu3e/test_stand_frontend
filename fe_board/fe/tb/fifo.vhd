library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity fifo is
    Generic (
           add  : natural := 8;
           width  : natural := 8
           );
    Port ( Din   : in  STD_LOGIC_VECTOR (width-1 downto 0);
           Wr    : in  STD_LOGIC;
           Dout  : out STD_LOGIC_VECTOR (width-1 downto 0);
           Rd    : in  STD_LOGIC;
           Empty : out STD_LOGIC;
           Full  : out STD_LOGIC;
           CLK   : in  STD_LOGIC;
           reset_n : in  STD_LOGIC
           );
end fifo;

architecture rtl of fifo is

signal wrcnt : std_logic_vector(add-1 downto 0);
signal rdcnt : std_logic_vector(add-1 downto 0);
type ram is array(0 to (2**add)-1) of std_logic_vector(width-1 downto 0);
signal memory : ram;
signal full_loc  : std_logic;
signal empty_loc : std_logic;

begin

  fifo_pro : process(reset_n, CLK)
  begin
  if(reset_n = '0')then
    wrcnt   <= (others => '0');
    rdcnt   <= (others => '0');
    Dout    <= (others => '0');
    memory    <= (others => (others => '0'));
  elsif(rising_edge(CLK)) then
     if (Wr='1' and full_loc='0') then
        memory(to_integer(unsigned(wrcnt))) <= Din;
        wrcnt <= wrcnt + '1';
     end if;
     if (Rd='1' and empty_loc='0') then
        Dout <= memory(to_integer(unsigned(rdcnt)));
        rdcnt <= rdcnt + '1';
     end if;
  end if;
  end process;

  full_loc  <= '1' when unsigned(rdcnt) = unsigned(wrcnt) + 1 else '0';
  empty_loc <= '1' when unsigned(rdcnt) = unsigned(wrcnt)   else '0';
  Full  <= full_loc;
  Empty <= empty_loc;

end rtl;