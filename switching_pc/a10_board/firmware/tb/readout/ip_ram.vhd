library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity ip_ram is
    Port ( clock   : in  STD_LOGIC;
           wren : in  STD_LOGIC;
           wraddress   : in  STD_LOGIC_VECTOR (11 downto 0);
           rdaddress   : in  STD_LOGIC_VECTOR (11 downto 0);
           data   : in  STD_LOGIC_VECTOR (31 downto 0);
           q  : out STD_LOGIC_VECTOR (31 downto 0)
         );
end ip_ram;

architecture BlockRAM of ip_ram is
type speicher is array(0 to (2**12)-1) of STD_LOGIC_VECTOR(31 downto 0);
signal memory : speicher;   
begin
  process begin
    wait until rising_edge(clock);
    if (wren='1') then
      memory(to_integer(unsigned(wraddress))) <= data;
    end if;
    q <= memory(to_integer(unsigned(rdaddress)));
  end process;
end BlockRAM;