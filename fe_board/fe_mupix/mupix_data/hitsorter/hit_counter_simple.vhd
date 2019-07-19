-----------------------------------
--
-- yet simpler hit counter
--
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity hit_counter_simple is
	PORT( 
		clock 		: 	in std_logic;
		reset_n 		: 	in std_logic;
		coarse_ena	: 	in std_logic;
		hits_ena_in	: 	in std_logic;
		counter		:	out std_logic_vector(47 downto 0)
	);
end hit_counter_simple;

architecture RTL of hit_counter_simple is

signal	hit_cnt 			: std_logic_vector(47 downto 0);
signal	hits_ena_reg 	: std_logic;

signal 	coarse_ena_reg	:  std_logic;

begin

counter 		<= hit_cnt;

	process(clock, reset_n)
	begin
		if(reset_n = '0')then
			hit_cnt				<= (others => '0');
			hits_ena_reg		<= '0';
			coarse_ena_reg		<= '0';
		elsif(rising_edge(clock))then
			coarse_ena_reg		<= coarse_ena;
			hits_ena_reg		<= hits_ena_in;
			if(hits_ena_reg = '1' and coarse_ena_reg ='0')then
				hit_cnt <= hit_cnt + 1;
			end if;
		end if;
	end process;

end RTL;