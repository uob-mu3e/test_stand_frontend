-- Sequencer_NG.vhd
-- Niklaus Berger, July 2021
-- niberger@uni-mainz.de
--
-- Take the front-stacked counters and bit arrays of busy timestamps to 
-- create a sequence of memory adrreses to be read and multiplexer settings
-- for the read part of the hit sorter

-- The ouput command has the TS in the LSBs, followed by four bits hit address
-- four bits channel/chip ID and the MSB inciating command (1) or hit (0)


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.numeric_std.all;

use work.mupix.all;


entity sequencer_ng is 
	port (
		reset_n							: in std_logic;										-- async reset
		clk								: in std_logic;										-- clock
		from_fifo						: sorterfifodata_t;
		fifo_empty						: in std_logic;
		read_fifo						: out std_logic;
		outcommand						: out command_t;
		command_enable					: out std_logic;
		outoverflow						: out std_logic_vector(15 downto 0)
		);
end sequencer_ng;

architecture rtl of sequencer_ng is

signal running: std_logic;
signal running_last: std_logic;
type output_type is(none, header1, header2, subheader, hits, footer);
signal output: output_type;
signal current_block:	block_t;
constant block_max:		block_t := (others => '1');
signal current_ts:		ts_t;
signal counters_reg: std_logic_vector(MEMCOUNTERRANGE);
signal subaddr:			counter_t;
signal dohits: std_logic;
signal from_fifo_reg	: sorterfifodata_t;
signal read_fifo_int: std_logic;
signal fifo_reg_valid: std_logic;
signal fifo_new: std_logic;
signal fifo_empty_last : std_logic;
signal read_token :std_logic;

begin

read_fifo <= read_fifo_int;

process(reset_n, clk)
	variable stop_fifo_reading : std_logic;
begin
if (reset_n = '0') then	
	running 		<= '0';
	running_last 	<= '0';
	output 			<= none;
	dohits			<= '0';
	read_fifo_int	<= '0';
	fifo_reg_valid	<= '0';
	fifo_new		<= '0';
	fifo_empty_last	<= '1';
	read_token		<= '1';
elsif (clk'event and clk = '1') then
	stop_fifo_reading := '0';
	running_last 	<= running;
	if (running = '0')then
		if (fifo_empty = '0') then
			running <= '1';
		end if;
	else
		if (running_last = '0' and running = '1') then
			output <= header1;
		elsif (output = header1) then
			output <= header2;
			if (from_fifo(HASMEMBIT) = '1') then
				stop_fifo_reading := '1';
			end if;
		elsif (output = header2) then
			output <= subheader;
			current_block <= from_fifo(TSBLOCKINFIFORANGE);
			if (from_fifo(HASMEMBIT) = '1') then
				stop_fifo_reading := '1';
				counters_reg <= from_fifo(MEMCOUNTERRANGE);
				current_ts	 <= from_fifo(TSINFIFORANGE);
				fifo_new	 <= '0';
			end if;
		elsif (output = footer) then
			output <= header1;
			stop_fifo_reading	:= '1';
		elsif (dohits = '1') then
			stop_fifo_reading	:= '1';
			if(counters_reg(3 downto 0) = "0001") then -- switch chip
				counters_reg(counters_reg'left-8 downto 0)	 <= counters_reg(counters_reg'left downto 8);
				counters_reg(counters_reg'left downto counters_reg'left-7)	 <= (others => '0');
				subaddr					<= "0000";
			else -- more hits from same chip
				counters_reg <= counters_reg;
				counters_reg(3 downto 0) <= counters_reg(3 downto 0) -'1';
				subaddr					 <= subaddr + "1";
			end if;
			if((counters_reg(3 downto 0) = "0010" and counters_reg(11 downto 8) = "0000") or
				(counters_reg(3 downto 0) = "0001" and counters_reg(11 downto 8) = "0001" and counters_reg(19 downto 16) = "0000")) then
				dohits <= '0';
				stop_fifo_reading	:= '0';
			end if;		
		elsif (from_fifo(TSBLOCKINFIFORANGE) /= current_block and fifo_new = '1') then
			output 			<= subheader;
			current_block 	<= from_fifo(TSBLOCKINFIFORANGE);
			if(from_fifo(HASMEMBIT) = '1')then -- could get faster here by not stopping the read if there is a single
				stop_fifo_reading := '1';
				fifo_new	 	<= '1';
				counters_reg 	<= from_fifo(MEMCOUNTERRANGE);
				current_ts	 	<= from_fifo(TSINFIFORANGE);
			else
				fifo_new	 	<= '0';
			end if;
			if(current_block = block_max) then
				output 			<= footer;
				stop_fifo_reading := '1';
				fifo_new	 	<= '1';
			end if;
		elsif (from_fifo(HASMEMBIT) = '1' and fifo_new = '1') then
			output 			<= hits;
			counters_reg	<= from_fifo(MEMCOUNTERRANGE);
			current_ts	 	<= from_fifo(TSINFIFORANGE);
			fifo_new	 	<= '0';
			subaddr			<= "0000";
			stop_fifo_reading := '0';
			dohits			<= '1';
			if(from_fifo(3 downto 0) = "0001" and from_fifo(11 downto 8) = "0000") then
				dohits <= '0';
			end if;
		elsif (from_fifo(HASMEMBIT) = '0' and fifo_new = '1') then
			output 			<= none;
			fifo_new	 	<= '0';
		else -- fifo empty
			output 			<= none;
		end if;
	end if;

	read_fifo_int	<= '0';
	if(running = '1' and (stop_fifo_reading = '0' or read_token = '1'))  then
		read_fifo_int		<= '1';
		if(fifo_empty = '0') then
			read_token 	<= '0';
		else 
			read_token 	<= '1';
		end if;
	end if;

	if(read_fifo_int = '1' and fifo_empty = '0') then
		fifo_new	<= '1';
		read_fifo_int	<= '0';
	end if;
end if;
end process;

process(reset_n, clk)
begin
if(reset_n = '0') then	
	command_enable 	<= '0';
	outcommand		<= (others => '0');
elsif(clk'event and clk = '1') then
	case output is
		when none =>
			command_enable 	<= '0';
			outcommand		<= (others => '0');
		when header1 =>
			outcommand 		<= COMMAND_HEADER1;
			command_enable 	<= '1';
		when header2 =>
			outcommand 		<= COMMAND_HEADER2;
			command_enable 	<= '1';
		when subheader =>
			outcommand 					<= COMMAND_SUBHEADER;
			outcommand(TSRANGE)			<= current_block & conv_std_logic_vector(0, BITSPERTSBLOCK);
			command_enable 	<= '1';
		when hits =>
			command_enable 									 <= '1';
			outcommand(COMMANDBITS-1)						 <= '0'; -- Hits, not a command
			outcommand(TSRANGE)								 <= current_ts; 
			outcommand(COMMANDBITS-2 downto TIMESTAMPSIZE+4) <= counters_reg(7 downto 4);
			outcommand(COMMANDBITS-6 downto TIMESTAMPSIZE)   <= subaddr;
		when footer =>
			outcommand		<= COMMAND_FOOTER;
			command_enable 	<= '1';
	end case;
end if;
end process;

end architecture rtl;
