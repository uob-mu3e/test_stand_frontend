library IEEE;
use ieee.std_logic_unsigned.all;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ip_ram is
    Port ( clock   : in  STD_LOGIC;
           read_clock : in  STD_LOGIC;
           wren : in  STD_LOGIC;
           wraddress   : in  STD_LOGIC_VECTOR (10 downto 0);
           rdaddress   : in  STD_LOGIC_VECTOR (10 downto 0);
           data   : in  STD_LOGIC_VECTOR (255 downto 0);
           q  : out STD_LOGIC_VECTOR (255 downto 0)
         );
end ip_ram;

architecture BlockRAM of ip_ram is
type speicher is array(0 to (2**12)-1) of STD_LOGIC_VECTOR(255 downto 0);
signal memory : speicher;
signal last_read_add :  STD_LOGIC_VECTOR (10 downto 0);
begin
  process(clock)
  begin
  if rising_edge(clock) then
    if (wren='1') then
      memory(to_integer(unsigned(wraddress))) <= data;
    end if;
  end if;
  end process;


  process(read_clock)
  begin
  if rising_edge(read_clock) then
    last_read_add <= rdaddress;
    q <= memory(to_integer(unsigned(last_read_add)));
  end if;
  end process;
end BlockRAM;