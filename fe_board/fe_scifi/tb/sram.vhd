library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
--use ieee.std_logic_unsigned.all;

entity sram is
  port (
    clock          : in  std_logic;
    reset_n        : in  std_logic;
    we             : in  std_logic;
    read_address   : in  std_logic_vector(15 downto 0);
    write_address  : in  std_logic_vector(15 downto 0);
    datain         : in  std_logic_vector(31 downto 0);
    dataout        : out std_logic_vector(31 downto 0)
  );
end entity sram;

architecture RTL of sram is

   type ram_type is array (15 downto 0) of std_logic_vector(31 downto 0);
   signal ram : ram_type := (others=>(others=>'0'));

begin

  RamProc: process(clock, reset_n) is
  begin
    if (reset_n = '0') then
      dataout <= (others => '0');
    elsif rising_edge(clock) then
      if we = '1' then
        ram(to_integer(unsigned(write_address))) <= datain;
      end if;
      dataout <= ram(to_integer(unsigned(read_address)));
    end if;
  end process RamProc;


end architecture RTL;