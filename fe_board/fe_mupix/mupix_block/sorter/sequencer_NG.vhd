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
signal counters_reg: std_logic_vector(MEMCOUNTERRANGE);
signal subaddr:			counter_t;
signal dohits: std_logic;
signal from_fifo_reg	: sorterfifodata_t;
signal fifo_empty_reg: std_logic;

begin


process(reset_n, clk)
	variable do_fifo_reading : std_logic;
begin
if (reset_n = '0') then	
	running 		<= '0';
	running_last 	<= '0';
	output 			<= none;
	read_fifo 		<= '0';
	fifo_empty_reg	<= '1';
	dohits			<= '0';
elsif (clk'event and clk = '1') then
	do_fifo_reading := '0';
	read_fifo 		<= '0';
	running_last 	<= running;
	if (running = '0')then
		if (fifo_empty = '0') then
			running <= '1';
		end if;
	else
		if (running_last = '0' and running = '1') then
			output <= header1;
			do_fifo_reading := '1';
		elsif (output = header1) then
			output <= header2;
		elsif (output = header2) then
			output <= subheader;
			current_block <= from_fifo_reg(TSBLOCKINFIFORANGE);
			if (from_fifo_reg(HASMEMBIT) = '0') then
				do_fifo_reading := '1';
			end if;
		elsif (output = footer) then
			output <= header1;
		elsif (dohits = '1') then
			if(counters_reg(3 downto 0) = "0001") then -- switch chip
				counters_reg(counters_reg'left-8 downto 0)	 <= counters_reg(counters_reg'left downto 8);
				counters_reg(counters_reg'left downto counters_reg'left-7)	 <= (others => '0');
				subaddr					<= "0000";
			else -- more hits from same chip
				counters_reg <= counters_reg;
				counters_reg(3 downto 0) <= counters_reg(3 downto 0) -'1';
				subaddr					 <= subaddr + "1";
			end if;
			if(counters_reg(3 downto 0) = "0001" and counters_reg(11 downto 8) = "0000") then
				dohits <= '0';
			end if;
		elsif (from_fifo_reg(TSBLOCKINFIFORANGE) /= current_block) then
			output 			<= subheader;
			do_fifo_reading := '1';
			current_block <= from_fifo_reg(TSBLOCKINFIFORANGE);
			if(current_block = block_max) then
				output 			<= footer;
				do_fifo_reading := '0';
			end if;
		elsif (from_fifo(HASMEMBIT) = '1' and fifo_empty_reg = '0') then
			output 			<= hits;
			counters_reg	<= from_fifo_reg(MEMCOUNTERRANGE);
			subaddr			<= "0000";
			do_fifo_reading := '1';
			dohits			<= '1';
		elsif (from_fifo_reg(HASMEMBIT) = '0' and fifo_empty_reg = '0') then
			output 			<= none;
			do_fifo_reading := '1';
		else -- fifo empty
			output 			<= none;
		end if;
	end if;

	if(do_fifo_reading = '1') then
		from_fifo_reg	<= from_fifo;
		read_fifo		<= '1';
		fifo_empty_reg	<= fifo_empty;
	end if;	
	if(fifo_empty_reg = '1' and fifo_empty = '0') then
		from_fifo_reg	<= from_fifo;
		read_fifo		<= '1';
		fifo_empty_reg	<= fifo_empty;
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
		when hits =>
			command_enable 									 <= '1';
			outcommand(COMMANDBITS-1)						 <= '0'; -- Hits, not a command
			outcommand(TSRANGE)								 <= current_block; 
			outcommand(COMMANDBITS-2 downto TIMESTAMPSIZE+4) <= counters_reg(7 downto 4);
			outcommand(COMMANDBITS-6 downto TIMESTAMPSIZE)   <= subaddr;
		when footer =>
			outcommand		<= COMMAND_FOOTER;
			command_enable 	<= '1';
	end case;
end if;
end process;

end architecture rtl;
