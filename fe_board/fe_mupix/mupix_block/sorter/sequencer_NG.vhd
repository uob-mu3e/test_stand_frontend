-- Sequencer_NG.vhd
-- Niklaus Berger, July 2021
-- niberger@uni-mainz.de
--
-- Take the front-stacked counters and bit arrays of busy timestamps to 
-- create a sequence of memory adrreses to be read and multiplexer settings
-- for the read part of the hit sorter

-- The ouput command has the TS in the LSBs, followed by four bits hit address
-- four bits channel/chip ID and the MSB inciating command (1) or hit (0)

-- In a bit more detail: For every TS which is either containg hits or marking the 
-- change of (16 TS) block, the main sorter file writes to a FIFO in the format

-- Timestamp | non-empty flag (hasmem)| Overflow bit|...|...|Chip Nr y | Nhits Chip y | Chip Nr x | Nhits Chip x
-- Note that the counters are right-stacked, i.e. the rightmost counter is from the first chip with hits and so on
-- If there are no more hits, the counter is 0

-- The FIFO is of the conventional (non show-ahead) type
-- It is almost equally tricky to make sure the FIFO is read whenever possible in showahead and non-showahead mode
-- After the integration run I decided that this way is easier to reason about

-- This input is then turned into a series of commands corresponding directly to the output of the sorter, i.e.
-- Header 1
-- Header 2
-- Subheader
-- Hit
-- Hit
-- ...
-- Subheader
-- ...
-- ...
-- Footer
--
-- For the hits we output in which memory (i.e. chip) they are at which position (TS plus hit number)
--
-- Has to be reset between runs


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
		runend							: in std_logic;
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
signal stopped:	std_logic;
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
signal hasoverflow:		std_logic;
signal fifo_empty_last:	std_logic;
signal fifo_new: 		std_logic;
signal read_fifo_int: 	std_logic;
signal make_header:		std_logic_vector(1 downto 0);
signal blockchange:		std_logic;
signal no_copy_next:	std_logic;

signal overflowts 		: std_logic_vector(15 downto 0);
signal overflow_to_out  : std_logic_vector(15 downto 0);

begin

read_fifo <= read_fifo_int;

process(reset_n, clk)
	variable copy_fifo : std_logic;
begin
if (reset_n = '0') then	
	running 		<= '0';
	running_last 	<= '0';
	stopped			<= '0';
	read_fifo_int	<= '0';
	fifo_empty_last	<= '1';
	output 			<= none;
	fifo_new		<= '0';
	current_block	<= block_max;
	no_copy_next	<= '0';
	overflowts		<= (others => '0');
	make_header 	<= "00";
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
		output 			<= header1;
		make_header 	<= "01";
	elsif(make_header = "01")then
		output 			<= header2;
		make_header 	<= "00";
		no_copy_next	<= '0';
	else
		if(blockchange = '1') then
			output		<= subheader;
			copy_fifo		:= '0';
			no_copy_next	<= '0';
			blockchange 	<= '0';
			overflow_to_out		<= overflowts;
			overflowts			<= (others => '0');
			if(hasmem = '0' and hasoverflow = '1') then
				overflowts		<= (others => '1'); -- Note that overflow gets sent with the next subheader!
			end if;
			-- Why 11 downto 8? We should be able to remove that
			if(counters_reg(3 downto 0) = "0000" and counters_reg(11 downto 8) = "0000")then
				copy_fifo				:= '1';
			end if;
		elsif(hasmem = '1')then
			output 			<= hits;
			copy_fifo		:= '0';
			overflowts(conv_integer(current_ts(TSINBLOCKRANGE))) <= hasoverflow;
			if(counters_reg(3 downto 0) = "0001" and counters_reg(11 downto 8) = "0000")then
				hasmem					<= '0';
			end if;

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
			if(counters_reg(3 downto 0) = "0001" and counters_reg(11 downto 8) = "0000") then
				copy_fifo				:= '1';
			end if;
		else
			output			<= none;
			copy_fifo		:= '1';

			if(hasmem = '0' and hasoverflow = '1')then
				overflowts(conv_integer(current_ts(TSINBLOCKRANGE))) <= hasoverflow;
			end if;
		end if;
	end if;	

	-- run end:
	if(stopped = '1') then
		output <= none;
	elsif(runend = '1' and current_block = block_max and fifo_empty = '1' and fifo_empty_last = '1' and stopped = '0') then
		output <= footer;
		stopped <= '1';
	end if;


	if(no_copy_next = '1')then
		copy_fifo := '0';
	end if;

	-- When to continue reading the FIFO
	if(from_fifo(HASMEMBIT) = '0' or fifo_empty = '1' or (from_fifo(3 downto 0) = "0001" and from_fifo(11 downto 8) = "0000")
		or copy_fifo = '1') then
		read_fifo_int <= '1';
	else
		read_fifo_int <= '0';
	end if;


	-- fifo_new means that there is valid output at the FIFO, which has not yet been copied
	-- to the respective variables
	-- copy_fifo means that the current set of variables was processed and they can be replaced
	-- with the fifo output
	--if(read_fifo_int = '1' and fifo_empty_last = '0')then
	if(fifo_empty_last = '0')then
		if(copy_fifo = '1')then
			fifo_new		<= '0';
		else
			fifo_new		<= '1';
		end if;	
	elsif(copy_fifo = '1' and fifo_new = '1')then
		fifo_new		<= '0';
	end if;	
	--if(copy_fifo = '1' and ((read_fifo_int = '1' and fifo_empty_last = '0') or fifo_new = '1'))then
	if(copy_fifo = '1' and ((fifo_empty_last = '0') or fifo_new = '1'))then
		current_block 	<= from_fifo(TSBLOCKINFIFORANGE);
		current_ts	 	<= from_fifo(TSINFIFORANGE);
		counters_reg	<= from_fifo(MEMCOUNTERRANGE);
		hasmem 			<= from_fifo(HASMEMBIT);
		hasoverflow		<= from_fifo(MEMOVERFLOWBIT);
		subaddr			<= "0000";
		if(from_fifo(HASMEMBIT)='1' and from_fifo(3 downto 0) = "0000")then --there was an overflow, no hits for this TS
			hasmem			<= '0';
			hasoverflow		<= '1';
		end if;
		if(from_fifo(TSBLOCKINFIFORANGE) /= current_block)then
			blockchange <= '1';
			if(from_fifo(HASMEMBIT)='1' and from_fifo(3 downto 0) /= "0000") then
				no_copy_next <= '1';
				read_fifo_int <= '0';
			end if;
			if(from_fifo(TSBLOCKINFIFORANGE) = block_zero)then
				make_header  <= "1" & running_last; -- this ensures that we do not output a footer at run start
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
	outoverflow		<= (others => '0');
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
			outoverflow					<= overflow_to_out;
			command_enable 				<= '1';
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
