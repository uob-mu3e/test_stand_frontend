-----------------------------------------------------------------------------
-- DDR3 Memory controller
--
-- Niklaus Berger, JGU Mainz
-- niberger@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity ddr3_memory_controller is 
port (
		reset_n	: in std_logic;
		
		-- Control and status registers
		ddr3control		: in std_logic_vector(15 downto 0);
		ddr3status		: out std_logic_vector(15 downto 0);
		
		ddr3clk			: out std_logic;
		ddr3_calibrated: out std_logic;
		ddr3_ready		: out std_logic;
		
		ddr3addr			: in std_logic_vector(25 downto 0);
		ddr3datain		: in std_logic_vector(511 downto 0);
		ddr3dataout		: out std_logic_vector(511 downto 0);
		ddr3_write		: in std_logic;
		ddr3_read		: in std_logic;
		ddr3_read_valid: out std_logic;
		
		-- Error counters
		errout			: out reg32;

		-- IF to DDR3 
		M_cal_success	:	in std_logic;
		M_cal_fail		:	in	std_logic;
		M_clk				:	in	std_logic;
		M_reset_n		:	in	std_logic;
		M_ready			:	in	std_logic;
		M_read			:	out std_logic;
		M_write			:	out std_logic;
		M_address		:  out std_logic_vector(25 downto 0);
		M_readdata		:	in	std_logic_vector(511 downto 0);
		M_writedata		:	out std_logic_vector(511 downto 0);
		M_burstcount	:	out std_logic_vector(6 downto 0);
		M_readdatavalid:  in std_logic
);
end entity;

architecture RTL of ddr3_memory_controller is

	type controller_mode_type is (disabled, waiting, ready, countertest, dataflow);
	signal mode : controller_mode_type;
	
	signal counter_read				:	std_logic;
	signal counter_write				:	std_logic;
	signal counter_address			:  std_logic_vector(25 downto 0);
	constant counter_address_1		:  std_logic_vector(25 downto 0) := (others => '1');
	signal counter_writedata		:	std_logic_vector(511 downto 0);
	signal counter_readdata			:	std_logic_vector(511 downto 0);
	signal counter_readdatavalid	:  std_logic;
	signal counter_readdatavalid_reg	:  std_logic;
		
	type counter_state_type is (disabled, writing, reading, done);
	signal counter_state : counter_state_type;
	signal mycounter 	: reg32;
	
	signal RWDone			: std_logic;
	
	signal Running		: std_logic;
	
	signal LastC		: std_logic_vector(27 downto 0);
	
	signal check		: std_logic_vector(15 downto 0);
	signal pcheck		: std_logic_vector(15 downto 0);
	
	signal poserr_reg		: reg32;
	signal counterr_reg	: reg32;	
	signal timecount_reg	: reg32;
	
	signal started 	: std_logic;
	
begin
		
		ddr3clk			<= M_clk;
		ddr3_calibrated<= M_cal_success;
		ddr3_ready		<= M_ready;


	-- Counter forwarding
	errout		<= poserr_reg 		when ddr3control(DDR3_COUNTERSEL_RANGE_A) = "01" else			
						counterr_reg	when ddr3control(DDR3_COUNTERSEL_RANGE_A) = "10" else
						timecount_reg;

	-- Status register
	ddr3status(DDR3_BIT_CAL_SUCCESS)	<= M_cal_success;
	ddr3status(DDR3_BIT_CAL_FAIL)	<= M_cal_fail;
	ddr3status(DDR3_BIT_RESET_N)	<= M_reset_n;
	ddr3status(DDR3_BIT_READY)	<= M_ready;
	
	ddr3status(DDR3_BIT_TEST_WRITING)	<= '1' when counter_state = writing else
							'0';
	ddr3status(DDR3_BIT_TEST_READING)	<= '1' when counter_state = reading else
							'0';	
	ddr3status(DDR3_BIT_TEST_DONE)	<= '1' when counter_state = done else
							'0';							

-- Mode MUX	
M_read	<= counter_read when mode = countertest else
				ddr3_read    when mode = dataflow		else
				'0';
				
M_write	<= counter_write when mode = countertest else
				ddr3_write    when mode = dataflow	 else
				'0';

M_address <= counter_address when mode = countertest else
				 ddr3addr    when mode = dataflow;
				
M_writedata	<= counter_writedata when mode = countertest else
					ddr3datain    when mode = dataflow;	
					
					
ddr3_read_valid	<= M_readdatavalid when mode = dataflow else '0';
ddr3dataout			<= M_readdata;			

-- This is the HW burst size...					
M_burstcount <= "0000001";					


					 
-- Mode state machine
process(M_clk, reset_n)
begin
if(reset_n = '0') then
	mode <= disabled;
elsif(M_clk'event and M_clk='1') then	
	case mode is
		when disabled =>
			if(ddr3control(DDR3_BIT_ENABLE_A) = '1') then
				mode <= waiting;
			end if;
		when waiting =>
			if(M_reset_n = '1' and M_cal_success = '1') then
				mode <= ready;
			end if;
		when ready =>
			if (ddr3control(DDR3_BIT_COUNTERTEST_A) = '1') then
				mode <= countertest;
			else 
				mode <= dataflow;
			end if;
		when countertest =>
			if (ddr3control(DDR3_BIT_COUNTERTEST_A)='0') then
				mode <= ready;
			end if;
		when dataflow =>
			if (ddr3control(DDR3_BIT_COUNTERTEST_A)='1') then
				mode <= ready;
			end if;
	end case;
end if;
end process;	


	
-- counter test state machine
process(M_clk, reset_n)
	variable counter_var : reg32;
begin
if(reset_n = '0') then
	counter_state	 <= disabled;
	
	counter_read 	<= '0';
	counter_write 	<= '0';

	
elsif(M_clk'event and M_clk='1') then
	
	counter_read 	<= '0';
	counter_write 	<= '0';
	
	-- Register once to ease timing
	counter_readdata 			<= M_readdata;
	counter_readdatavalid 	<= M_readdatavalid;
	
	counter_readdatavalid_reg 		<= counter_readdatavalid;
	case counter_state is
		when disabled =>
			if (mode = countertest and M_ready = '1') then
				counter_state <= writing;
			end if;
			mycounter 				<= (others => '1');
			counter_address		<= (others => '1');
			RWDone <= '0';
			Running <= '0';
			
			poserr_reg		<= (others => '0');	
			counterr_reg	<= (others => '0');
			timecount_reg 	<= (others => '0');
			
			started 			<= '0';
			
		when writing =>
		
			timecount_reg <= timecount_reg + '1';
			
			counter_write <= '1';
			
			if (M_ready = '1') then
				started		<= '1';
				counter_var := mycounter + '1';
				mycounter <= mycounter + '1';
				counter_address <= counter_address + '1';
			else
				counter_var := mycounter;
			end if;
		
			counter_writedata <= X"1" & counter_var(27 downto 0) &
										X"2" & counter_var(27 downto 0) &
										X"3" & counter_var(27 downto 0) &
										X"4" & counter_var(27 downto 0) &
										X"5" & counter_var(27 downto 0) &
										X"6" & counter_var(27 downto 0) &
										X"7" & counter_var(27 downto 0) &
										X"8" & counter_var(27 downto 0) &
										X"9" & counter_var(27 downto 0) &
										X"A" & counter_var(27 downto 0) &
										X"B" & counter_var(27 downto 0) &
										X"C" & counter_var(27 downto 0) &
										X"D" & counter_var(27 downto 0) &
										X"E" & counter_var(27 downto 0) &
										X"F" & counter_var(27 downto 0) &
										X"0" & counter_var(27 downto 0);
										  				  
										  

			
			if(counter_address = counter_address_1 and started = '1') then
				RWDone <= '1';
				counter_write <= '0';
			end if;
			
			if(RWDone = '1')then
				counter_write <= '0';
				counter_state <= reading;
				counter_address		<= (others => '1');
				RWDone <= '0';
				started <= '0';
			end if;
			
		when reading =>
		
			timecount_reg <= timecount_reg + '1';	
			counter_read <= '1';

			
			if (M_ready = '1') then
				counter_address <= counter_address + '1';
				started         <= '1';
			end if;
			
			if(counter_address = counter_address_1 and started = '1') then
				RWDone <= '1';
				counter_read <= '0';
			end if;
			
			if(RWDone = '1')then
				counter_state <= done;
			end if;
			
			pcheck <= (others => '0');	
			check <= (others => '0');
			
			if (counter_readdatavalid = '1') then
				Running <= '1';
				LastC		<= counter_readdata(27 downto 0) + '1';
				if (Running = '1') then		
					if(counter_readdata(31 downto 28) /= X"0") then
						pcheck(0) <= '1';
					end if;
					if(counter_readdata(63 downto 60) /= X"F") then
						pcheck(1) <= '1';
					end if;
					if(counter_readdata(95 downto 92) /= X"E") then
						pcheck(2) <= '1';
					end if;
					if(counter_readdata(127 downto 124) /= X"D") then
						pcheck(3) <= '1';
					end if;
					if(counter_readdata(159 downto 156) /= X"C") then
						pcheck(4) <= '1';
					end if;
					if(counter_readdata(191 downto 188) /= X"B") then
						pcheck(5) <= '1';
					end if;
					if(counter_readdata(223 downto 220) /= X"A") then
						pcheck(6) <= '1';
					end if;
					if(counter_readdata(255 downto 252) /= X"9") then
						pcheck(7) <= '1';
					end if;
					if(counter_readdata(287 downto 284) /= X"8") then
						pcheck(8) <= '1';
					end if;
					if(counter_readdata(319 downto 316) /= X"7") then
						pcheck(9) <= '1';
					end if;
					if(counter_readdata(351 downto 348) /= X"6") then
						pcheck(10) <= '1';
					end if;
					if(counter_readdata(383 downto 380) /= X"5") then
						pcheck(11) <= '1';
					end if;
					if(counter_readdata(415 downto 412) /= X"4") then
						pcheck(12) <= '1';
					end if;
					if(counter_readdata(447 downto 444) /= X"3") then
						pcheck(13) <= '1';
					end if;
					if(counter_readdata(479 downto 476) /= X"2") then
						pcheck(14) <= '1';
					end if;
					if(counter_readdata(511 downto 508) /= X"1")	then
						pcheck(15) <= '1';
					end if;
					--------------------------------------------
					if(counter_readdata(27 downto 0) /= LastC) then
						check(0) <= '1';
					end if;
					if(counter_readdata(59 downto 32) /= LastC) then
						check(1) <= '1';
					end if;
					if(counter_readdata(91 downto 64) /= LastC) then
						check(2) <= '1';
					end if;
					if(counter_readdata(123 downto 96) /= LastC) then
						check(3) <= '1';
					end if;
					if(counter_readdata(155 downto 128) /= LastC) then
						check(4) <= '1';
					end if;
					if(counter_readdata(187 downto 160) /= LastC) then
						check(5) <= '1';
					end if;
					if(counter_readdata(219 downto 192) /= LastC) then
						check(6) <= '1';
					end if;
					if(counter_readdata(251 downto 224) /= LastC) then
						check(7) <= '1';
					end if;
					if(counter_readdata(283 downto 256) /= LastC) then
						check(8) <= '1';
					end if;
					if(counter_readdata(315 downto 288) /= LastC) then
						check(9) <= '1';
					end if;
					if(counter_readdata(347 downto 320) /= LastC) then
						check(10) <= '1';
					end if;
					if(counter_readdata(379 downto 352) /= LastC) then
						check(11) <= '1';
					end if;
					if(counter_readdata(411 downto 384) /= LastC) then
						check(12) <= '1';
					end if;
					if(counter_readdata(443 downto 416) /= LastC) then
						check(13) <= '1';
					end if;
					if(counter_readdata(475 downto 448) /= LastC) then
						check(14) <= '1';
					end if;
					if(counter_readdata(507 downto 480) /= LastC) then
						check(15) <= '1';
					end if;
				end if;
			end if;
			if(counter_readdatavalid_reg = '1') then
				if(pcheck /= "0000000000000000")then
					poserr_reg <= poserr_reg + '1';
				end if;
				if(check /= "0000000000000000")then
					counterr_reg <= counterr_reg + '1';
				end if;
			end if;
		when done =>
			if (mode /= countertest) then
				counter_state <= disabled;
			end if;
	end case;
end if;
end process;

end architecture;
