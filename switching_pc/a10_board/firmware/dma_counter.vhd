-- simple counter for pcie test
-- counter_test.vhd

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_unsigned.all;
use IEEE.numeric_std.all;

entity dma_counter is
	generic (
		bits            : integer := 256--;
	);
	port(
		i_clk 			: 	in STD_LOGIC;
		i_reset_n		:	in std_logic;
		i_enable		:	in std_logic;
		i_dma_wen_reg	:	in std_logic;
		i_fraccount     :	in std_logic_vector(7 downto 0);
		o_dma_wen		:	out std_logic;
		o_cnt 			: 	out STD_LOGIC_VECTOR (bits - 1 downto 0)--;
		);
end dma_counter;

architecture arch of dma_counter is

	signal timer 		: 	STD_LOGIC_VECTOR (bits - 1 downto 0);
	signal counter 		: 	std_logic_vector(7 downto 0);
	
begin

process(i_clk, i_reset_n)
begin
if(i_reset_n = '0') then
	o_dma_wen	<= '0';
	timer		<= (others => '1');
	counter   	<= (others => '0');
	o_cnt   	<= (others => '0');
elsif(rising_edge(i_clk)) then
	if(i_enable = '0') then
		o_dma_wen 	<= '0';
		timer		<= (others => '0');
		counter   	<= (others => '0');
		o_cnt   	<= (others => '0');
	else
		counter 	<= counter + '1';
		if(counter <= i_fraccount) then
			o_dma_wen 	<= i_dma_wen_reg;
			timer 		<= timer + '1';
		end if;
		o_cnt 	<= timer;
	end if;
end if;
end process;

end arch;	