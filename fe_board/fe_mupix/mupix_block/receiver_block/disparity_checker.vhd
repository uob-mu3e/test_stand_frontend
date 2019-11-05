-----------------------------------
--
-- 8b10b disparity checker
-- S. Dittmeier
-- September 2017
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity disparity_checker is 
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		rx_in					: in std_logic_vector (9 DOWNTO 0);	
		ready					: in std_logic;
		disp_err				: out std_logic
		);
end disparity_checker;

architecture RTL of disparity_checker is

	signal running_disp 	: std_logic_vector(1 downto 0);
	signal ready_reg	  	: std_logic_vector(3 downto 0);
	signal error 			: std_logic;
	signal ones 			: std_logic_vector(3 downto 0);

begin
	

	process( clk)
	variable count : unsigned(3 downto 0) := x"0";
	begin
   count := x"0";   --initialize count variable.
	if(rising_edge(clk))then
		 for i in 0 to 9 loop   --check for all the bits.
			  if(rx_in(i) = '1') then --check if the bit is '1'
					count := count + 1; --if its one, increment the count.
			  end if;
		end loop;
		ones <= std_logic_vector(count);    --assign the count to vector.
	end if;
	end process;
	
	process(reset_n, clk)
	begin
		if(reset_n = '0')then
		
			error 			<= '0';
			disp_err 		<= '0';
			running_disp 	<= "00";
			ready_reg		<= (others => '0');
			
		elsif(rising_edge(clk))then
			
			ready_reg 		<= ready_reg(2 downto 0) & ready;
			error 	<= '0';
			if(ready_reg(2) = '1')then
				if(ones = x"4")then
					running_disp <= running_disp - 1;
				elsif(ones = x"5")then
					running_disp <= running_disp;
				elsif(ones = x"6")then
					running_disp <= running_disp + 1;
				else
					error	<= '1'; -- indicating hard 8b10b error!
				end if;
			else
				running_disp	<= running_disp;
			end if;
			
			disp_err 	<= '0';	
			-- running disparity states, starting from "00" neutral
			-- "11" means - 1
			-- "01" means + 1
			-- "10" means +2 or -2: so error state
			if((running_disp = "10" or error = '1') and ready_reg(3)='1')then
				disp_err 		<= '1';
				running_disp	<= "00";	-- we start over with disparity check in raw state
			end if;
			
		end if;
	end process;

end RTL;