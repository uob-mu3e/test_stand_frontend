library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


entity max10_dev is
port (

	CLOCK : in std_logic;
	LED1 : out std_logic--;
	
);
end entity;

architecture arch of max10_dev is

	signal counter : std_logic_vector(31 downto 0) := (others => '0');
	
begin

	LED1 <= counter(20);

	process(CLOCK)
	begin
		if(rising_edge(CLOCK)) then
			counter <= counter + 1;
		end if;
	end process;
	
end architecture;	