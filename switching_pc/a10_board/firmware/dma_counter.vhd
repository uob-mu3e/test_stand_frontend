-- simple counter for pcie test
-- counter_test.vhd

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_unsigned.all;
use IEEE.numeric_std.all;

entity dma_counter is
port (
		i_clk 					:	in std_logic;
		i_reset_n				:	in std_logic;
		i_enable					:	in std_logic;
		i_dma_wen_reg			:	in std_logic;
		i_fraccount     		:	in std_logic_vector(7 downto 0);
		i_halffull_mode		:	in std_logic;
		i_dma_halffull			:	in std_logic;
		o_dma_end_event		:	out std_logic;
		o_dma_wen				:	out std_logic;
		o_cnt 					: 	out std_logic_vector (159 downto 0)--;
);
end entity;

architecture arch of dma_counter is

	signal timer 		: 	std_logic_vector (31 downto 0);
	signal counter 		: 	std_logic_vector(7 downto 0);
	signal dma_halffull_cnt : std_logic_vector(63 downto 0);
	signal dma_not_halffull_cnt : std_logic_vector(63 downto 0);
	
begin

process(i_clk, i_reset_n)
begin
if(i_reset_n = '0') then
	o_dma_wen			<= '0';
	o_dma_end_event	<= '0';
	timer				<= (others => '0');
	counter   			<= (others => '0');
	o_cnt   			<= (others => '0');
	dma_halffull_cnt <= (others => '0');
	dma_not_halffull_cnt <= (others => '0');
elsif(rising_edge(i_clk)) then
	if(i_enable = '0') then
		o_dma_wen 			<= '0';
		o_dma_end_event	<= '0';
		timer				<= (others => '0');
		counter   			<= (others => '0');
		o_cnt   			<= (others => '0');
		dma_halffull_cnt <= (others => '0');
		dma_not_halffull_cnt <= (others => '0');
	else
		counter <= counter + '1';
		o_dma_end_event <= '0';
		if(i_halffull_mode = '0') then
			o_cnt(31 downto 0)  <= timer;
			o_cnt(159 downto 31) <= (others => '0');
			if(counter <= i_fraccount) then						
				if(timer(7 downto 0) = x"FF") then -- if 2^4*256=4kb is written 
					o_dma_end_event <= '1';
				end if;
				o_dma_wen 	<= i_dma_wen_reg;
				timer 		<= timer + '1';
			else
				o_dma_wen 	<= '0';
			end if;
		else
			if(counter <= i_fraccount) then
				o_cnt(31 downto 0)  <= timer;
				o_cnt(95 downto 32) <= dma_halffull_cnt;
				o_cnt(159 downto 96) <= dma_not_halffull_cnt;
				if (i_dma_halffull = '1') then
					o_dma_wen <= '0';
					dma_halffull_cnt <= dma_halffull_cnt + '1';
				else
					if(timer(7 downto 0) = x"FF") then -- if 2^4*256=4kb is written 
						o_dma_end_event <= '1';
					end if;
					timer <= timer + '1';
					o_dma_wen <= i_dma_wen_reg;
					dma_not_halffull_cnt <= dma_not_halffull_cnt + '1';
				end if;
			else
					o_dma_wen <= '0';
			end if;	
		end if;
	end if;
end if;
end process;

end architecture;
