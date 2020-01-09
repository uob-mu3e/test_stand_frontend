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

	signal running_disp : std_logic_vector(1 downto 0);
	signal ready_reg	  : std_logic_vector(3 downto 0);
	
	type add_array_1 is array (3 downto 0) of std_logic_vector(1 downto 0);
	type add_array_2 is array (1 downto 0) of std_logic_vector(2 downto 0);
	
	signal add_stage_1 : add_array_1;
	signal add_stage_2 : add_array_2;
	signal add_stage_3 : std_logic_vector(3 downto 0);
	
	signal error : std_logic;
	
begin
	
	-- we make use of two embedded adders inside one ALM
	-- which can do two two-bit additions and two three-bit additions without additional resources

gen_adds:
for i in 0 to 1 generate

	i_adder_1_lo : entity work.add_1_bit
	port map(
		clk	=> clk,
		x		=> rx_in(i*3),
		y		=> rx_in(i*3+1),
		cin	=> rx_in(i*3+2),
		sum	=> add_stage_1(i)(0),
		cout	=> add_stage_1(i)(1)
	);
	
	i_adder_1_hi : entity work.add_1_bit
	port map(
		clk	=> clk,
		x		=> rx_in(i*2+6),
		y		=> rx_in(i*2+7),
		cin	=> '0',
		sum	=> add_stage_1(i+2)(0),
		cout	=> add_stage_1(i+2)(1)
	);	
	
	i_adder_2 : entity work.add_2_bits
	PORT map
	(
		clock		=> clk,
		dataa		=> add_stage_1(i),
		datab		=> add_stage_1(i+2),
		cout		=> add_stage_2(i)(2),
		result	=> add_stage_2(i)(1 downto 0)
	);
	
end generate;

	i_adder_3 : entity work.add_3_bits
	PORT map
	(
		clock		=> clk,
		dataa		=> add_stage_2(0),
		datab		=> add_stage_2(1),
		cout		=> add_stage_3(3),
		result	=> add_stage_3(2 downto 0)
	);
	
	process(reset_n, clk)
	begin
		if(reset_n = '0')then
		
			error 			<= '0';
			disp_err 		<= '0';
			running_disp 	<= "00";
			ready_reg		<= (others => '0');
--			add_stage_1		<= (others => (others => '0'));
--			add_stage_2		<= (others => (others => '0'));
--			add_stage_3		<= (others => '0');
			
		elsif(rising_edge(clk))then
			
			ready_reg 		<= ready_reg(2 downto 0) & ready;
		
--			add_stage_1(0) <= rx_in(2) + rx_in(1) + rx_in(0);	-- max 3
--			add_stage_1(1) <= rx_in(5) + rx_in(4) + rx_in(3);
--			add_stage_1(2) <= rx_in(7) + rx_in(6);					-- max 2
--			add_stage_1(3) <= rx_in(9) + rx_in(8);
--			add_stage_2(0) <= add_stage_1(0) + add_stage_1(2);	-- max 5
--			add_stage_2(1) <= add_stage_1(1) + add_stage_1(3);		
--			add_stage_3 	<= add_stage_2(0) + add_stage_2(1);	-- max 10
			
			
			error 	<= '0';
			if(ready_reg(2) = '1')then
				if(add_stage_3 = x"4")then
					running_disp <= running_disp - 1;
				elsif(add_stage_3 = x"5")then
					running_disp <= running_disp;
				elsif(add_stage_3 = x"6")then
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

end architecture;
