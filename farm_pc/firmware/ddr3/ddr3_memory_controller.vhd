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
use work.ddr3_components.all;
use work.pcie_components.all;


entity ddr3_memory_controller is 
	port (
		clk		: in std_logic;  -- clock for the FPGA fabric side
		reset_n	: in std_logic;
		
		-- Control and status registers
		ddr3control		: in reg32;
		ddr3status		: out reg32;
		ddr3addr			: in reg32;
		ddr3datain		: in reg32;
		ddr3dataout		: out reg32;
		
		-- IF to DDR3 A
		A_cal_success	:	in std_logic;
		A_cal_fail		:	in	std_logic;
		A_clk				:	in	std_logic;
		A_reset			:	in	std_logic;
		A_ready			:	in	std_logic;
		A_read			:	out std_logic;
		A_write			:	out std_logic;
		A_address:		:  out std_logic_vector(25 downto 0);
		A_readdata		:	in	std_logic_vector(511 downto 0);
		A_writedata		:	out std_logic_vector(511 downto 0);
		A_burstcount	:	out std_logic_vector(6 downto 0);
		A_readdatavalid:  out std_logic;
		
		-- IF to DDR3 B
 		B_cal_success	:	in std_logic;
		B_cal_fail		:	in	std_logic;
		B_clk				:	in	std_logic;
		B_reset			:	in	std_logic;
		B_ready			:	in	std_logic;
		B_read			:	out std_logic;
		B_write			:	out std_logic;
		B_address:		:  out std_logic_vector(25 downto 0);
		B_readdata		:	in	std_logic_vector(511 downto 0);
		B_writedata		:	out std_logic_vector(511 downto 0);
		B_burstcount	:	out std_logic_vector(6 downto 0);
		B_readdatavalid:  out std_logic
	);
end entity ddr3_memory_controller;
	
architecture RTL of ddr3_memory_controller is

	type controller_mode_type is (disabled, waiting, ready, countertest, pcietest);
	signal mode : controller_mode_type;


begin

	ddr3status(0)	<= A_cal_success;
	ddr3status(1)	<= A_cal_fail;
	ddr3status(2)	<= A_reset;
	ddr3status(3)	<= A_ready;
	
	ddr3status(4)	<= B_cal_success;
	ddr3status(5)	<= B_cal_fail;
	ddr3status(6)	<= B_reset;
	ddr3status(7)	<= B_ready;
	
process(clk, reset_n)

begin

if(reset_n = '0') then
	mode <= disabled;
elsif(clk'event and clk='1') then
	
	case mode is
		when disabled =>
			if(ddr3control(0) = 1) then
				mode <= waiting;
			end if;
		when waiting =>
		
		when ready =>
		
		when countertest =>
		
		when pcietest =>
		
	end case;
end if;
end process;	



end architecture RTL;