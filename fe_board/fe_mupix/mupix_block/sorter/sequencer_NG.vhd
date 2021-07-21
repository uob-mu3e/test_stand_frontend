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
constant block_zero:	block_t := (others => '0');
signal current_ts:		ts_t;
signal ts_to_out:		ts_t;
signal counters_reg: std_logic_vector(MEMCOUNTERRANGE);
signal subaddr:			counter_t;
signal subaddr_to_out:  counter_t;
signal chip_to_out:		counter_t;
signal hasmem:			std_logic;
signal fifo_empty_last:	std_logic;
signal fifo_new: 		std_logic;
signal read_fifo_int: 	std_logic;
signal make_header:		std_logic_vector(1 downto 0);
signal blockchange:		std_logic;
signal no_copy_next:	std_logic;

begin

read_fifo <= read_fifo_int;

process(reset_n, clk)
	variable copy_fifo : std_logic;
begin
if (reset_n = '0') then	
	running 		<= '0';
	running_last 	<= '0';
	read_fifo_int	<= '0';
	fifo_empty_last	<= '1';
	output 			<= none;
	fifo_new		<= '0';
	current_block	<= block_max;
	no_copy_next	<= '0';

elsif (clk'event and clk = '1') then

	
	running_last 	<= running;
	if (running = '0')then
		if (fifo_empty = '0') then
			running <= '1';
		end if;
	end if;


	fifo_empty_last	<= fifo_empty;

	copy_fifo	:= '1';

	ts_to_out		<= current_ts;

	if(make_header = "11")then
		output 			<= footer;
		make_header 	<= "10";
	elsif(make_header = "10")then
		output <= header1;
		make_header 	<= "01";
	elsif(make_header = "01")then
		output <= header2;
		make_header 	<= "00";
		no_copy_next	<= '0';
	else
		if(blockchange = '1')then
			output 			<= subheader;
			blockchange 	<= '0';
		elsif(hasmem = '1')then
			output 			<= hits;
			copy_fifo		:= '0';
			if(counters_reg(3 downto 0) = "0001") then -- switch chip
				counters_reg(counters_reg'left-8 downto 0)	 <= counters_reg(counters_reg'left downto 8);
				counters_reg(counters_reg'left downto counters_reg'left-7)	 <= (others => '0');
				subaddr					<= "0000";
				subaddr_to_out			<= subaddr;	
				chip_to_out				<=  counters_reg(7 downto 4);
			else -- more hits from same chip
				counters_reg(3 downto 0) <= counters_reg(3 downto 0) -'1';
				subaddr					 <= subaddr + "1";
				subaddr_to_out			 <= subaddr;
				chip_to_out				<=  counters_reg(7 downto 4);
			end if;
		else
			output			<= none;
		end if;
		if(counters_reg(3 downto 0) = "0001" and counters_reg(11 downto 8) = "0000")then
			hasmem					<= '0';
			copy_fifo				:= '1';
		end if;
	end if;		

	-- When to continue reading the FIFO
	if(from_fifo(HASMEMBIT) = '0' or fifo_empty = '1' or (from_fifo(3 downto 0) = "0001" and from_fifo(11 downto 8) = "0000"))then
		read_fifo_int <= '1';
	else
		read_fifo_int <= '0';
	end if;

	if(no_copy_next = '1')then
		copy_fifo := '0';
	end if;

	
	if(read_fifo_int = '1' and fifo_empty_last = '0')then
		if(copy_fifo = '1')then
			fifo_new		<= '0';
		else
			fifo_new		<= '1';
		end if;	
	elsif(copy_fifo = '1' and fifo_new = '1')then
		fifo_new		<= '0';
	end if;	

	if(copy_fifo = '1' and ((read_fifo_int = '1' and fifo_empty_last = '0') or fifo_new = '1'))then
		current_block 	<= from_fifo(TSBLOCKINFIFORANGE);
		current_ts	 	<= from_fifo(TSINFIFORANGE);
		counters_reg	<= from_fifo(MEMCOUNTERRANGE);
		hasmem 			<= from_fifo(HASMEMBIT);
		subaddr			<= "0000";
		if(from_fifo(TSBLOCKINFIFORANGE) /= current_block)then
			blockchange <= '1';
			if(from_fifo(TSBLOCKINFIFORANGE) = block_zero)then
				make_header <= "1" & running_last;
				no_copy_next <= '1';
			end if;
		else
			blockchange <= '0';
		end if;
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
			outcommand(TSBLOCKRANGE)	<= ts_to_out(TSBLOCKRANGE);
			outcommand(TSNONBLOCKRANGE)	<= (others => '0');
			command_enable 	<= '1';
		when hits =>
			command_enable 									 <= '1';
			outcommand(COMMANDBITS-1)						 <= '0'; -- Hits, not a command
			outcommand(TSRANGE)								 <= ts_to_out; 
			outcommand(COMMANDBITS-2 downto TIMESTAMPSIZE+4) <= chip_to_out;
			outcommand(COMMANDBITS-6 downto TIMESTAMPSIZE)   <= subaddr_to_out;
		when footer =>
			outcommand		<= COMMAND_FOOTER;
			command_enable 	<= '1';
	end case;
end if;
end process;

end architecture rtl;
