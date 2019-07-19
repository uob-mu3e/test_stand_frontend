--------------------------------------------------------
-- Hitsorter for MuPix8	 										--
-- based on hitsorter for MuPix7 and multisorter		--
-- mupix data streams											--
--------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use IEEE.numeric_std.all;

use work.hitsorter_components.all;
use work.mupix_constants.all;
use work.mupix_types.all;

----------------------------------
-- To do list:
-- * chip marker (so far it is the link number)
-- * MuPix7 telescope had number of hits and the binary counter in the header. Do we want this for MPx8 as well?
----------------------------------

entity hitsorter is
	generic(
		NINPUT:	integer := 1;	-- 1 FPGA has 16 inputs, i.e. 4 inputs to the sorter with a 4to1Mux before the sorter
		NSORTERINPUTS:	integer := 1
----		NINPUT2:	integer := 1; -- half NINPUT or 1 (whichever is bigger)
----		NINPUT4:	integer := 1; -- one quarter NINPUT or 1 (whichever is bigger)
----		NINPUT8:  integer := 1; -- one eight NINPUT or 1 (whichever is bigger)
----		NINPUT16:  integer := 1;-- one sixteenth NINPUT or 1 (whichever is bigger)
----		HITSIZE: integer := 2+2+8+8+6+10	-- serializer + link + Row + Col + Charge + TS
--		
	);
	port (
		reset_n	:	in	std_logic;
		writeclk: 	in	std_logic;
		tsdivider:	in std_logic_vector(3 downto 0);
		readclk:		in	std_logic;
		
		hits_ser:			in hit_array_t;	--4x40b
		hits_ser_en:		in std_logic_vector(NINPUT-1 downto 0);
		ts_en:		in std_logic_vector(NINPUT-1 downto 0);
		
		tsdelay:		in	std_logic_vector(TSRANGE);
		
		fromtrigfifo: 	in reg64;
		trigfifoempty:	in std_logic;	
		fromhbfifo: 	in reg64;
		hbfifoempty: 	in std_logic;	
		
		data_out:	out std_logic_vector(31 downto 0);
		out_ena:		out std_logic;
		out_eoe:		out std_logic;
		
		out_hit:		out std_logic;		-- for hit counter
		
		readtrigfifo:	out std_logic;
		readhbfifo: 	out std_logic;	
		
		received_hits:		out std_logic_vector(31 downto 0);	
		outoftime_hits:	out std_logic_vector(31 downto 0);	
		intime_hits:		out std_logic_vector(31 downto 0);	
		memwrite_hits:		out std_logic_vector(31 downto 0);	
		overflow_hits:		out std_logic_vector(31 downto 0);	
		sent_hits:			out std_logic_vector(31 downto 0);
		break_counter:		out std_logic_vector(31 downto 0)
	);
end hitsorter;	


architecture rtl of hitsorter is

	signal tsdivcounter:	std_logic_vector(3 downto 0);
	signal ts_local:		std_logic_vector(TSRANGE);
	signal ts_shifted:	std_logic_vector(TSRANGE);
--	signal ts_shifted2:	std_logic_vector(TSRANGE);
	signal fpga_ts:		reg62;
	signal writeclk_ts:	reg62;
	
	signal hitblockcounter	:	std_logic_vector(30 downto 0);
	signal sentblockcounter	: 	std_logic_vector(30 downto 0);
	
	subtype addr_t 				is std_logic_vector(12 downto 0);
	type addr_array_t				is array (NINPUT-1 downto 0) of addr_t;
	subtype counter_t 			is std_logic_vector(2 downto 0);
	type counter_array_t			is array (2**10-1 downto 0) of counter_t;
	type all_counter_array_t	is array (NINPUT-1 downto 0) of counter_array_t;
	type block_counter_array_t	is array (2**7-1 downto 0) of counter_t;
	type all_block_counter_array_t	is array (NINPUT-1 downto 0) of block_counter_array_t;
	constant counter_zero:		std_logic_vector := "000";
	subtype memhit_t				is std_logic_vector((MEMHITSIZE-10)-1 downto 0);
	type memhit_array_t			is array (NINPUT-1 downto 0) of memhit_t;
	
	signal wcounters:				all_counter_array_t;
	signal rcounters:				all_block_counter_array_t;	-- all_counter_array_t;
	
	subtype indicator_array_t 	is std_logic_vector(2**10-1 downto 0);
	subtype block_indicator_array_t is std_logic_vector(7 downto 0);
	type all_indicator_array	is array (NINPUT-1 downto 0) of indicator_array_t;
	subtype perblock_indicator_array_t is std_logic_vector(2**7-1 downto 0);
	type all_perblock_indicator_array_t is array (NINPUT-1 downto 0) of perblock_indicator_array_t;
	
	signal w_empty_n:				all_indicator_array;
	signal w_overflow:			all_indicator_array;
	signal r_overflow:			all_perblock_indicator_array_t;	-- all_indicator_array;
	
	signal empty_n:				indicator_array_t;
	signal block_empty_n:		block_indicator_array_t;
	signal r_block_empty_n:		block_indicator_array_t;
	signal empty_n_32:			std_logic_vector(31 downto 0);
	signal empty_n_8:				std_logic_vector(7 downto 0);
	
	signal read_empty_n:			indicator_array_t;
	signal r_empty_n:				all_perblock_indicator_array_t;	-- all_indicator_array;
	
--	signal overflow:				indicator_array_t;
--	signal block_overflow:		block_indicator_array_t;
--	signal overflow_32:			std_logic_vector(31 downto 0);
--	signal overflow_8:			std_logic_vector(7 downto 0);

	subtype errcounter_t is		std_logic_vector(31 downto 0);
	type all_errcounter_array_t	 is array (NINPUT-1 downto 0) of errcounter_t;
	
	signal receivedhits:			all_errcounter_array_t;
	signal outoftimehits:		all_errcounter_array_t;
	signal intimehits:			all_errcounter_array_t;
	signal memwritehits:       all_errcounter_array_t;
	signal overflowhits:       all_errcounter_array_t;
	signal senthits:				errcounter_t;
	
	subtype block_t				is std_logic_vector(2 downto 0);
	subtype BLOCKRANGE			is integer range 9 downto 7;
	subtype NOTBLOCKRANGE		is integer range 6 downto 0;
	type block_array_t			is array (NINPUT-1 downto 0) of block_t;
	
	signal counterblock:			block_t;
	signal writeblock:			block_array_t;
	signal sumblock:				block_t;
	signal rblock1:				block_t;
	signal rblock2:				block_t;
	signal resetblock:			block_t;
	signal sumblock_del:			block_t;
	signal sumblock_del2:		block_t;
	signal sumblock_del3:		block_t;
	signal sumblock_del4:		block_t;
	signal sumblock_del5:		block_t;
	
	signal waddrs:					addr_array_t;
	signal wens:					std_logic_vector(NINPUT-1 downto 0);
	signal wensmem:				std_logic_vector(2*NINPUT-1 downto 0);
	signal whits:					memhit_array_t;
	subtype MEMHITRANGE			is integer range 39 downto 10;
	
	signal raddrs:					addr_array_t;
	signal rhits:					memhit_array_t;
	
	-- summing is currently not used
--	subtype csum2_t 			is std_logic_vector(3 downto 0);
--	type csum2_array_t			is array (2**7-1 downto 0) of csum2_t;
--	type all_csum2_array_t		is array (NINPUT2-1 downto 0) of csum2_array_t;
--	signal csum2:				all_csum2_array_t;
--	
--	subtype csum4_t 			is std_logic_vector(4 downto 0);
--	type csum4_array_t			is array (2**7-1 downto 0) of csum4_t;
--	type all_csum4_array_t		is array (NINPUT4-1 downto 0) of csum4_array_t;
--	signal csum4:				all_csum4_array_t;
--	
--	subtype csum8_t 			is std_logic_vector(5 downto 0);
--	type csum8_array_t			is array (2**7-1 downto 0) of csum8_t;
--	type all_csum8_array_t		is array (NINPUT8-1 downto 0) of csum8_array_t;
--	signal csum8:				all_csum8_array_t;
--	
--	subtype csum16_t 			is std_logic_vector(6 downto 0);
--	type csum16_array_t			is array (2**7-1 downto 0) of csum16_t;
--	type all_csum16_array_t		is array (NINPUT16-1 downto 0) of csum16_array_t;
--	signal csum16:					all_csum16_array_t;
--	signal csum16final:			all_csum16_array_t;
--	
--	type fullcsum16_array_t		is array (2**10-1 downto 0) of csum16_t;
--	signal wsum16:				fullcsum16_array_t;
	
	signal readsumblocktmp:		block_t;
	signal readsumblocktmp1:	block_t;
	signal readsumblocktmp2:	block_t;
	signal readsumblocklast:	block_t;
	signal readsumblockl2:		block_t;
	signal readsumblock:			block_t;
	signal readsumblock2:		block_t;	-- duplicate
	
	signal nextblock			:	block_t;
	signal newblock			:	std_logic;
	signal gotoend				:	std_logic;
--	signal gotoend2			:	std_logic;
	signal waitforblock		:	std_logic;
--	signal startreadout		: 	std_logic;
	
	signal blockchange			:	std_logic;
	signal break					:	std_logic;
	signal break_counter_reg	: 	std_logic_vector(31 downto 0);
	
	type readstate_type is (IDLE, STARTREADOUT, HEADER1, HEADER2, HEADER3, HEADER4, READING1, READING2, FOOTER, BEGINTRIGGER, TRIGGERHEADER, TRIGGERHEADER2, TRIGGERDATA_MSB, TRIGGERDATA_LSB, TRIGGERFOOTER, BEGINHB, HBHEADER, HBDATA_MSB, HBDATA_LSB, HBFOOTER);
	signal readstate : readstate_type;
	signal currentblock: block_t;
	signal currentts : std_logic_vector(NOTBLOCKRANGE);
	signal currentts2 : std_logic_vector(NOTBLOCKRANGE);	-- just a duplicate
	signal currentts3 : std_logic_vector(NOTBLOCKRANGE);	-- just a duplicate
	signal nextts : std_logic_vector(NOTBLOCKRANGE);
	signal previousts : std_logic_vector(NOTBLOCKRANGE);
	
	signal data2			: std_logic_vector(31 downto 0);
		
	-- for triggers/hitbus
	signal trigtoggle 		: std_logic;
	signal triggercounter	: reg32;
	signal hbcounter		 	: reg32;
	
	signal is_reading			: std_logic;

signal hits_en				: STD_LOGIC_VECTOR(NSORTERINPUTS-1 downto 0);
signal hits					: hit_array_t;
	
	
begin

process(writeclk)
begin
	if(rising_edge(writeclk)) then
		hits_en	<= hits_ser_en and (not ts_en);
		hits		<= hits_ser;
	end if;
end process;

-- Generate local and shifted copy of timestamp
process(writeclk, reset_n)
begin
if(reset_n = '0') then
	ts_local 		<= (others => '0');
	ts_shifted		<= (others => '0');
--	ts_shifted2		<= (others => '0');
	tsdivcounter	<= (others => '0');
	sumblock			<= (others => '0');
	rblock1			<= (others => '0');
	rblock2			<= (others => '0');
	resetblock		<= (others => '0');
	counterblock	<= (others => '0');
	fpga_ts			<= (others => '0');
	writeclk_ts		<= (others => '0');
elsif(writeclk'event and writeclk = '1') then	
	tsdivcounter 	<= tsdivcounter + '1';
	if(tsdivcounter = tsdivider) then
		tsdivcounter	<= (others => '0');
		ts_local			<= ts_local + '1';
		fpga_ts			<= fpga_ts + '1';
	end if;
	writeclk_ts		<= writeclk_ts	+ '1';
	ts_shifted 		<= ts_local + tsdelay;
--	ts_shifted2		<= ts_shifted;					-- TODO: Do we really need ts_shifted2? -- to be used in case of timing problems
	
	sumblock			<= ts_shifted(BLOCKRANGE) + "100";
	rblock1			<= ts_shifted(BLOCKRANGE) + "011";
	rblock2			<= ts_shifted(BLOCKRANGE) + "010";
	resetblock		<= ts_shifted(BLOCKRANGE) + "001";
	counterblock	<= ts_shifted(BLOCKRANGE);
end if;
end process;


-- Write process
genwrite: for i in NINPUT-1 downto 0 generate
	process(writeclk, reset_n)
	begin
	if(reset_n = '0') then
		wens(i)				<= '0';
		wcounters(i)		<= (others => counter_zero);
		w_empty_n(i)		<= (others => '0');
		w_overflow(i)		<= (others => '0');
		outoftimehits(i)	<= (others => '0');
		intimehits(i)		<= (others => '0');
		overflowhits(i)	<= (others => '0');
		memwritehits(i)	<= (others => '0');
		receivedhits(i)	<= (others => '0');
		writeblock(i)		<= (others => '0');
	elsif(writeclk'event and writeclk = '1') then
		wens(i) <= '0';
		if (hits_en(i) = '1') then				-- there are hits available
			receivedhits(i)	<= receivedhits(i) + '1';
			writeblock(i)		<= hits(i)(BLOCKRANGE);
			if(hits(i)(BLOCKRANGE) = sumblock or			-- check if hits are in the writing range
				hits(i)(BLOCKRANGE) = rblock1  or
				hits(i)(BLOCKRANGE) = rblock2  or
				hits(i)(BLOCKRANGE) = resetblock) then
				
				outoftimehits(i) <= outoftimehits(i) + '1';
				
			else	-- hits are in the writing range 
				if(w_overflow(i)(conv_integer(hits(i)(TSRANGE))) = '0') then	-- check if we are not yet in overflow
					wens(i)		<= '1';
					whits(i)		<= hits(i)(MEMHITRANGE);
					waddrs(i)	<= hits(i)(TSRANGE) & wcounters(i)(conv_integer(hits(i)(TSRANGE)));
					w_empty_n(i)(conv_integer(hits(i)(TSRANGE))) <= '1';	-- indicate that there are hits with this TS written to memory
		
					-- if logic utilization becomes a problem, think about writing the w_counters to memory as well.
					if(wcounters(i)(conv_integer(hits(i)(TSRANGE))) = "000" and w_empty_n(i)(conv_integer(hits(i)(TSRANGE))) = '1') then	-- we have already written 8 entries for this TS -> overflow, disable writing
						w_overflow(i)(conv_integer(hits(i)(TSRANGE))) <= '1';
						overflowhits(i) <= overflowhits(i) + '1';
						wens(i)	<= '0';
					else		-- we can still write to this address. prepare next counter, i.e. wcounters++. 
								-- this counter is ahead of the actual number of entries by 1. 
								-- need to take care of this when copying counters in read process
						wcounters(i)(conv_integer(hits(i)(TSRANGE))) 	<= wcounters(i)(conv_integer(hits(i)(TSRANGE))) + '1';
						memwritehits(i) <= memwritehits(i) + '1';							
					end if;
				else		-- we are in overlow
					overflowhits(i) <= overflowhits(i) + '1';
				end if;
				intimehits(i)	<= intimehits(i) + '1';
				
			end if; -- hit in/out of time
		end if; -- hits_en
		
		-- Reset counters etc.
		for j in 2**7-1 downto 0 loop
			wcounters(i)( conv_integer(resetblock & "0000000") + j) <= (others => '0');
         w_empty_n(i)( conv_integer(resetblock & "0000000") + j) <= '0';
         w_overflow(i)(conv_integer(resetblock & "0000000") + j) <= '0';
		end loop;
--		-- shouldn't this work as well?
--		wcounters(i)( conv_integer(resetblock & "1111111") downto conv_integer(resetblock & "0000000")) <= (others => (others => '0'));
--		w_empty_n(i)( conv_integer(resetblock & "1111111") downto conv_integer(resetblock & "0000000")) <= (others => '0');
--		w_overflow(i)(conv_integer(resetblock & "1111111") downto conv_integer(resetblock & "0000000")) <= (others => '0');
		
	end if;	-- reset_n
	end process;
end generate genwrite;

-- Check full/empty per timeslice and block, sum over hits
process(writeclk, reset_n)
	variable empty_n_tmp : indicator_array_t;
--	variable overflow_tmp : indicator_array_t;
begin
if(reset_n = '0') then
	
	empty_n				<= (others => '0');
	block_empty_n		<= (others => '0');
	
	sumblock_del		<= (others => '0');
	sumblock_del2		<= (others => '0');
	sumblock_del3		<= (others => '0');
	sumblock_del4		<= (others => '0');
	sumblock_del5		<= (others => '0');
	
	empty_n_32	<= (others => '0');
	empty_n_8	<= (others => '0');

	
elsif(writeclk'event and writeclk = '1') then	

	empty_n_tmp		:= (others => '0');
--	overflow_tmp	:= (others => '0');
	
	for i in NINPUT-1 downto 0 loop
		for j in 2**10-1 downto 0 loop
		
			if w_empty_n(i)(j) = '1' then
				empty_n_tmp(j) := '1';		-- empty_n_tmp indicates that there were hits with this TS at at least one of the inputs
			end if;
		
		end loop;
	end loop;
		
	-- potentially restrict this to the first two or so cycles of sumblock?
	block_empty_n(conv_integer(sumblock)) 	<= '0';
	
	-- copy information about not-empty for the current sumblock
	for i in 2**7-1 downto 0 loop 	
		empty_n(conv_integer(sumblock & "0000000")+i)	<= empty_n_tmp(conv_integer(sumblock & "0000000")+i);	
	end loop;
	-- build a per-block empty_n. this requires an OR over the 128 TS in the current sumblock. do it in 3 stages
	-- 128 to 32
	for j in 31 downto 0 loop
		for i in 3 downto 0 loop
			if empty_n(conv_integer(sumblock & "0000000")+4*j+i) = '1' then
				empty_n_32(j)	<= '1';
			end if;
		end loop;
	end loop;
	-- 32 to 8
	for j in 7 downto 0 loop
		for i in 3 downto 0 loop
			if empty_n_32(4*j+i) = '1' then
				empty_n_8(j)	<= '1';
			end if;
		end loop;
	end loop;
	-- 8 to 1
	for i in 7 downto 0 loop
		if empty_n_8(i) = '1' then
			block_empty_n(conv_integer(sumblock))	<= '1';
		end if;
	end loop;
	
	-- block_empty_n is ready 5 clk cyc after sumblock has changed, i.e. coincident with sumblock_del5
	sumblock_del	<= sumblock;
	sumblock_del2	<= sumblock_del;
	sumblock_del3	<= sumblock_del2;
	sumblock_del4	<= sumblock_del3;
	sumblock_del5	<= sumblock_del4;

	-- reset intermediate not-empty signals with a change of sumblock
	if ((sumblock = sumblock_del) and (sumblock_del /= sumblock_del2)) then
		empty_n_32	<= (others => '0');
		empty_n_8	<= (others => '0');
	end if;
	
	-- sum hits
	-- summing is not used so far
--	for i in 2**7-1 downto 0 loop
--		for j in NINPUT2-1 downto 0 loop
--			if(NINPUT = 1) then
--				csum2(j)(i) <= "0" & wcounters(j)(conv_integer(sumblock & "0000000")+i);
--			else
--				csum2(j)(i) <= ("0" & wcounters(j*2)(conv_integer(sumblock & "0000000")+i))
--							  + ("0" & wcounters(j*2+1)(conv_integer(sumblock & "0000000")+i));
--			end if;
--		end loop;
--		
--		for j in NINPUT4-1 downto 0 loop
--			if(NINPUT2 = 1) then
--				csum4(j)(i) <= "0" & csum2(j)(i);
--			else
--				csum4(j)(i) <= ("0" & csum2(2*j)(i)) + ("0" & csum2(2*j+1)(i));
--			end if;
--		end loop;
--		
--		for j in NINPUT8-1 downto 0 loop
--			if(NINPUT4 = 1) then
--				csum8(j)(i) <= "0" & csum4(j)(i);
--			else
--				csum8(j)(i) <= ("0" & csum4(2*j)(i)) + ("0" & csum4(2*j+1)(i));
--			end if;
--		end loop;
--		
--		for j in NINPUT16-1 downto 0 loop
--			if(NINPUT16 = 1) then
--				csum16(j)(i) <= "0" & csum8(j)(i);
--			else
--				csum16(j)(i) <= ("0" & csum8(2*j)(i)) + ("0" & csum8(2*j+1)(i));
--			end if;
--		end loop;		
--	end loop;
--	
--	if(ts_shifted(6 downto 0) = "0000100") then	-- if(ts_shifted2(6 downto 0) = "0000100") then		-- TODO: is this really dependent on ts_shifted2? or do we simply need to wait for 4 clock cycles?
--		csum16final	<= csum16;
--	end if;

end if;	-- reset_n
end process;


genmem: for i in NINPUT-1 downto 0 generate


smem : work.hitsortermem 
	PORT MAP
	(
		data				=> whits(i),
		rdaddress		=> raddrs(i),
		rdclock			=> readclk,
		wraddress		=> waddrs(i),
		wrclock			=> writeclk,
		wren				=> wens(i),
		q					=> rhits(i)
	);


end generate genmem;


-- Read process
read_proc: process(readclk, reset_n)
	variable n_in			: integer range NINPUT-1 downto 0 := 0;
	variable prev_n_in	: integer range NINPUT-1 downto 0 := 0;
	variable n_in_data	: integer range NINPUT-1 downto 0 := 0;
begin
if(reset_n = '0') then

	for i in NINPUT-1 downto 0 loop
		rcounters(i)		<= (others => counter_zero);
	end loop;
	
	for i in NINPUT-1 downto 0 loop
		raddrs(i)		<= (others => '0');
	end loop;
		
	readsumblocktmp	<= (others => '0'); 
	readsumblocktmp1	<= (others => '0'); 
	readsumblocktmp2	<= (others => '0'); 
	readsumblocklast 	<= (others => '0'); 
	readsumblockl2		<= (others => '0'); 
	readsumblock		<= (others => '0');	
	readsumblock2		<= (others => '0');	
	
	r_block_empty_n	<= (others => '0');
	
	currentblock		<= (others => '0');	
	currentts			<= (others => '0');	
	currentts2			<= (others => '0');	
	currentts3			<= (others => '0');	
	nextblock			<= (others => '0');	
	nextts				<= (others => '0');
	previousts			<= (others => '0');
	
	gotoend				<= '0';
--	gotoend2				<= '0';
	
	blockchange			<= '0';
	break					<= '0';
	break_counter_reg	<= (others => '0');
	
	hitblockcounter	<= (others => '0');
	sentblockcounter	<= (others => '0');
	
--	wsum16				<= (others => (others => '0'));
	
	readstate			<= IDLE;
	
	read_empty_n		<= (others => '0');
	r_empty_n			<= (others =>(others => '0'));
	r_overflow			<= (others =>(others => '0'));
	
	data_out				<= (others => '0');
	out_ena				<= '0';
	out_eoe				<= '0';
	
	out_hit				<= '0';
	
	waitforblock	<= '1';
	
	senthits			<= (others	=> '0');
	
	trigtoggle		<= '0';
	triggercounter	<= (others => '0');
	hbcounter		<= (others => '0');
	readtrigfifo	<= '0';
	readhbfifo		<= '0';
	
	n_in				:= 0;
	prev_n_in		:= 0;
	n_in_data		:= 0;
	
	is_reading		<= '0';
	
elsif(readclk'event and readclk = '1') then

	-- Hopefully clock transition safe determination of the current sum block on the 'other' clock
--	if (writeclk = '1') then
	readsumblocktmp2	<= sumblock_del5;
--	end if;
	readsumblocktmp1	<= readsumblocktmp2;
	readsumblocktmp	<= readsumblocktmp1;
	readsumblocklast	<= readsumblocktmp;
	readsumblockl2		<= readsumblocklast;
	
	-- indicate that we have moved with ts_shifted to the next block
	blockchange <= '0';	
	if((readsumblocktmp = readsumblocklast) and (readsumblockl2 /= readsumblocklast)) then
		blockchange			<= '1';
		hitblockcounter	<= hitblockcounter + '1';	-- this number is used in the header
	end if;

	if(blockchange = '1') then
		-- copy counters and not-empty for the readsumblock
		readsumblock		<= readsumblocklast;
		readsumblock2		<= readsumblocklast;
		
--		if (writeclk = '1') then
			for j in 2**7-1 downto 0 loop			
				r_block_empty_n(conv_integer(readsumblocklast))					<= block_empty_n(conv_integer(readsumblocklast));					-- not-empty per block, merge of all inputs
				read_empty_n(conv_integer(readsumblocklast & "0000000")+j)	<= empty_n(conv_integer(readsumblocklast & "0000000")+j);		-- not-empty per TS, merge of all inputs			
				-- summing not used so far
				-- wsum16(conv_integer(readsumblocklast & "0000000")+j) <= csum16final(0)(j); 	-- why only 0th component? (it should have only this component. it's the merge of up to 16 inputs) 
																															-- what's the purpose of wsum16?
																															-- TODO: it seems csum16final is not yet ready at this moment. has changed to some extent with introduction of sumblock_del2
			end loop;
--		end if;
	end if;
	
	-- Check that the current block is not outside of scope of the read side
	break	<= '0';
	if ( (currentblock /= readsumblock) and (currentblock /= readsumblock-"001") and (currentblock /= readsumblock-"010") and (currentblock /= readsumblock-"011") ) then 
	-- TODO: is it okay to read the reset block? Yes, as long as we don't allow it in the write process.
		break	<= '1';
	end if;
	
	-- Indicate that end of current block is reached
	newblock	<= '0';
	if read_empty_n(conv_integer(currentblock & nextts)) = '0' then
		newblock	<= '1';
	end if;
	
	gotoend	<= newblock;	-- indicate FSM to go to FOOTER state

	-- Determine next block and tell the FSM that there is a new block with data. Copy the counters for the next block.
	waitforblock	<= '1';
	if (newblock = '1') then	-- the current block is completely read	
		-- TODO: Q: also include reset block, i.e. readsumblock-3? A: did not work out well
--		if    ( r_block_empty_n(conv_integer(readsumblock-"011")) = '1' 
--					and not(readsumblock-"011" = currentblock) )	then -- corresponds to resetblock which is older than rblock2, rblock1 and sumblock
--			nextblock		<= readsumblock-"011";
--			waitforblock	<= '0';
--		elsif	( r_block_empty_n(conv_integer(readsumblock-"010")) = '1' 
		if	( r_block_empty_n(conv_integer(readsumblock-"010")) = '1'
--					and not(readsumblock-"011" = currentblock) 
					and not(readsumblock-"010" = currentblock) )	then -- corresponds to rblock2 which is older than rblock1 and sumblock
			nextblock		<= readsumblock-"010";
			waitforblock	<= '0';
			for j in 2**7-1 downto 0 loop
				for i in NINPUT-1 downto 0 loop
					rcounters(i)(j)	<= (wcounters(i)(conv_integer((readsumblock2-"010") & "0000000")+j)-'1');	-- counters per TS and input
					r_empty_n(i)(j)	<=  w_empty_n(i)(conv_integer((readsumblock -"010") & "0000000")+j);			-- not-empty per TS and input
					r_overflow(i)(j)	<= w_overflow(i)(conv_integer((readsumblock -"010") & "0000000")+j);			-- overflow per TS and input
				end loop;
			end loop;
		elsif ( r_block_empty_n(conv_integer(readsumblock-"001")) = '1' 
--					and not(readsumblock-"011" = currentblock) 
					and not(readsumblock-"010" = currentblock) 
					and not(readsumblock-"001" = currentblock) ) then	-- corresponds to rblock1 which is older than sumblock
			nextblock		<= readsumblock-"001";
			waitforblock	<= '0';
			for j in 2**7-1 downto 0 loop
				for i in NINPUT-1 downto 0 loop
					rcounters(i)(j)	<= (wcounters(i)(conv_integer((readsumblock2-"001") & "0000000")+j)-'1');	-- counters per TS and input
					r_empty_n(i)(j)	<=  w_empty_n(i)(conv_integer((readsumblock -"001") & "0000000")+j);			-- not-empty per TS and input
					r_overflow(i)(j)	<= w_overflow(i)(conv_integer((readsumblock -"001") & "0000000")+j);			-- overflow per TS and input
				end loop;
			end loop;
		elsif ( r_block_empty_n(conv_integer(readsumblock))       = '1' 
--					and not(readsumblock-"011" = currentblock) 
					and not(readsumblock-"010" = currentblock) 
					and not(readsumblock-"001" = currentblock) 
					and not(readsumblock = currentblock) )			then
			nextblock		<= readsumblock;
			waitforblock	<= '0';
			for j in 2**7-1 downto 0 loop
				for i in NINPUT-1 downto 0 loop
					rcounters(i)(j)	<= (wcounters(i)(conv_integer((readsumblock2) & "0000000")+j)-'1');	-- counters per TS and input
					r_empty_n(i)(j)	<=  w_empty_n(i)(conv_integer((readsumblock)  & "0000000")+j);			-- not-empty per TS and input
					r_overflow(i)(j)	<= w_overflow(i)(conv_integer((readsumblock)  & "0000000")+j);			-- overflow per TS and input
				end loop;
			end loop;
		end if;
	end if;
	
	-- Determine next TS
	for i in 2**7-1 downto 0 loop
		if i = currentts3 then
			exit;
		end if;
		if (read_empty_n(conv_integer(currentblock & "0000000")+i) = '1') then	-- have to exclude currentts as the reset of read_empty_n of this ts is delayed by one clk cycle
			nextts	<= std_logic_vector(to_unsigned(i,7));
		end if;
	end loop;
	
	-- build read address
	-- notice that all memories get the same read address, also if they might have no entries at this address.
	-- have to decide in the FSM which data output to take
	for i in NINPUT-1 downto 0 loop
		raddrs(i)(12 downto 10) 	<= currentblock;
		raddrs(i)( 9 downto  3) 	<= currentts;
		raddrs(i)( 2 downto  0)		<= rcounters(i)(conv_integer(currentts));
	end loop;
	
	out_ena	<= '0';
	out_eoe	<= '0';
	
	out_hit	<= '0';
	
	readtrigfifo	<= '0';
	readhbfifo		<= '0';

	----------------
	--- READ-FSM ---
	----------------
	case readstate is
	when IDLE	=>
		-- wait for a new block with data
		if (waitforblock = '0') then
			currentblock	<= nextblock;
		else
			currentblock	<= readsumblock-"011";	-- reset block
		end if;
		currentts		<= (others => '0');
		currentts2		<= (others => '0');
		currentts3		<= (others => '0');

		
		-- If there are trigger data, write them
		if(trigfifoempty = '0' and trigtoggle = '0') then
			readstate	<= TRIGGERHEADER;
			data_out		<= BEGINOFTRIGGER;
			out_ena		<= '1';
			readtrigfifo	<= '1';
		-- or hitbus events
		elsif(hbfifoempty = '0' and trigtoggle = '0')then
			readstate	<= HBHEADER;
			data_out		<= BEGINOFHB;
			out_ena		<= '1';
			readhbfifo	<= '1';			
		elsif (waitforblock = '0') then	-- a new block with data is available, start the readout sequence
			readstate  	<= STARTREADOUT;
		end if;
		trigtoggle	<= not trigtoggle;
		
	when STARTREADOUT	=>
		-- start of the readout sequence
		sentblockcounter	<= sentblockcounter + '1';	-- count number of blocks sent out
		data_out				<= BEGINOFEVENT;
		out_ena				<= '1';
		readstate    		<= HEADER1;
		
	when HEADER1 =>
		-- hit block counter
		if (read_empty_n(conv_integer(currentblock & "0000000")) = '1') then -- do not miss TS = 0
			currentts	<= (others	=> '0');
			currentts2	<= (others	=> '0');
			currentts3	<= (others	=> '0');
		else		
			currentts			<= nextts;
			currentts2			<= nextts;
			currentts3			<= nextts;
		end if;
		data_out				<= '0' & hitblockcounter;
		out_ena				<= '1';
		readstate			<= HEADER2;
		
	when HEADER2 =>
		-- MSBs of FPGA TS
		data_out		<= '0' & writeclk_ts(REG62_TOP_RANGE); -- TODO: do we want fpga_ts or writeclk_ts?
		out_ena		<= '1';
		for i in NINPUT-1 downto 0 loop							-- get the input with data at the current TS
			if r_empty_n(i)(conv_integer(currentts2)) = '1' then
				n_in	:= i;
			end if;
		end loop;
		prev_n_in	:= n_in;		-- needed when determining the next input
		readstate	<= HEADER3;
		
	when HEADER3 =>
		-- LSBs of FPGA TS
		data_out		<= '0' & writeclk_ts(REG62_BOTTOM_RANGE);
		out_ena		<= '1';
		readstate	<= HEADER4;

		n_in_data	:= n_in;		-- when we read the output of the memory, n_in has already moved so we need to keep this number
		if ( rcounters(n_in)(conv_integer(currentts)) = "000" ) then	-- if the current counter is 0, check, if there is another input with hits with the current TS
			for i in NINPUT-1 downto 0 loop
				if i = prev_n_in then
					exit;
				end if;
				if r_empty_n(i)(conv_integer(currentts2)) = '1' then
					n_in	:= i;
				end if;
			end loop;
			if n_in = prev_n_in then 																	-- if there is none, go on to the next TS
				currentts	<= nextts;
				currentts2	<= nextts;
				currentts3	<= nextts;
				read_empty_n(conv_integer(currentblock & currentts3))	<= '0';				-- the current TS for all inputs is now empty
				for i in NINPUT-1 downto 0 loop														-- get the input for the next TS
					if r_empty_n(i)(conv_integer(nextts)) = '1' then
						n_in	:= i;
					end if;
				end loop;
			end if;
			prev_n_in	:= n_in;
		else																									-- if the counter is not yet 0, simply decrement
			rcounters(n_in)(conv_integer(currentts))	<= rcounters(n_in)(conv_integer(currentts))-"001";
		end if;
		previousts	<= currentts2;

	when HEADER4 =>		-- could add overflow information here
		-- number of sent hit blocks	
		data_out		<= '0' & sentblockcounter;
		out_ena		<= '1';
		readstate	<= READING1;
		
	when READING1 =>
		-- first part of data word	
		data_out		<= "00" & rhits(n_in_data)(29 downto 26) & '0' & r_overflow(n_in_data)(conv_integer(previousts)) & HITLABEL  & rhits(n_in_data)(25 downto 22) & rhits(n_in_data)(21 downto 6);
		data2			<= "00" & rhits(n_in_data)(29 downto 26) & '0' & r_overflow(n_in_data)(conv_integer(previousts)) & TIMELABEL & rhits(n_in_data)(25 downto 22) & rhits(n_in_data)( 5 downto 0) & currentblock & previousts;
		out_ena		<= '1';
		
		out_hit		<= '1';

		n_in_data	:= n_in;
		if ( rcounters(n_in)(conv_integer(currentts)) = "000" ) then	-- if the current counter is 0, check, if there is another input with hits with the current TS
			for i in NINPUT-1 downto 0 loop
				if i = prev_n_in then
					exit;
				end if;
				if r_empty_n(i)(conv_integer(currentts2)) = '1' then
					n_in	:= i;
				end if;
			end loop;
			if n_in = prev_n_in then  																	-- if there is none, go on to the next TS
				currentts	<= nextts;
				currentts2	<= nextts;
				currentts3	<= nextts;
				read_empty_n(conv_integer(currentblock & currentts3))	<= '0';				-- the current TS for all inputs is now empty
				for i in NINPUT-1 downto 0 loop														-- get the input for the next TS
					if r_empty_n(i)(conv_integer(nextts)) = '1' then
						n_in	:= i;
					end if;
				end loop;
			end if;
			prev_n_in	:= n_in;
		else																									-- if the counter is not yet 0, simply decrement
			rcounters(n_in)(conv_integer(currentts))	<= rcounters(n_in)(conv_integer(currentts))-"001";
		end if;
		previousts	<= currentts2;

--		gotoend2		<=gotoend;		
		senthits		<= senthits+'1';
		readstate	<= READING2;
		
	when READING2 =>
		-- second part of data word
		data_out		<= data2;
		out_ena		<= '1';
		
		if break = '1' then
			break_counter_reg	<= break_counter_reg+'1';
		end if;
		
		if gotoend = '1' or break = '1' then	-- if the block is completely read or ran out of the reading range we have to finish this readout cycle
			readstate 	<= FOOTER;
			n_in			:= 0;
		else
			readstate	<= READING1;
		end if;
		
	when FOOTER =>
		data_out			<= ENDOFEVENT;
		out_ena			<= '1';
		out_eoe			<= '1';
		readstate 		<= IDLE;
		if (waitforblock = '0') then
			currentblock	<= nextblock;
		else
			currentblock	<= readsumblock-"011";	-- reset block
		end if;
		currentts		<= (others => '0');
		currentts2		<= (others => '0');
		currentts3		<= (others => '0');
		r_block_empty_n(conv_integer(currentblock))	<= '0';	-- the currentblock is now empty
		
		-- If there are trigger data, write them
		if(trigfifoempty = '0' and trigtoggle = '0') then
			readstate	<= BEGINTRIGGER;	-- added state to mitigate overwriting of ENDOFEVENT
--			readstate	<= TRIGGERHEADER;
--			data_out		<= BEGINOFTRIGGER;
--			out_ena		<= '1';
--			readtrigfifo	<= '1';
		-- or hitbus events
		elsif(hbfifoempty = '0' and trigtoggle = '0')then
			readstate	<= BEGINHB;		-- added state to mitigate overwriting of ENDOFEVENT
--			readstate	<= HBHEADER;
--			data_out		<= BEGINOFHB;
--			out_ena		<= '1';
--			readhbfifo	<= '1';			
		elsif (waitforblock = '0') then	-- if a new block with data is available, we can directly start a new readout sequence
			readstate  	<= STARTREADOUT;
		end if;
		trigtoggle	<= not trigtoggle;
		
	when BEGINTRIGGER =>
			readstate	<= TRIGGERHEADER;
			data_out		<= BEGINOFTRIGGER;
			out_ena		<= '1';
			readtrigfifo<= '1';
		
	when BEGINHB	=>
			readstate	<= HBHEADER;
			data_out		<= BEGINOFHB;
			out_ena		<= '1';
			readhbfifo	<= '1';			
		
		-- trigger events!	-- same as in the MuPix7 hitsorter_new	
	when TRIGGERHEADER =>
		-- count number of trigger blocks
		data_out			<= '0' & triggercounter(30 downto 0);
		out_ena			<= '1';
		triggercounter	<= triggercounter + '1';
		readtrigfifo 	<= '0';
		readstate		<= TRIGGERDATA_MSB;	
	when TRIGGERDATA_MSB =>
		-- Output the contents of the trigger FIFO until empty
		-- Note that this generates an extra read on the FIFO, which is why underflow protection is needed
		data_out			<= fromtrigfifo(REG64_TOP_RANGE);
		out_ena			<= '1';
		if(trigfifoempty = '1')then
			readtrigfifo <= '0';
		else
			readtrigfifo <= '1';
		end if;
		readstate		<= TRIGGERDATA_LSB;
		
	when TRIGGERDATA_LSB =>
		data_out			<= fromtrigfifo(REG64_BOTTOM_RANGE);
		out_ena			<= '1';
		readtrigfifo 	<= '0';						
		if(trigfifoempty = '1')then
			readstate 		<= TRIGGERFOOTER;
		else
			readstate 		<= TRIGGERDATA_MSB;						
		end if;		
		
	when TRIGGERFOOTER =>
		-- End of triggers, go back to start
		data_out 		<= ENDOFTRIGGER;
		out_ena			<= '1';
		out_eoe			<= '1';
		readstate		<= IDLE;
		if	waitforblock = '0' then		-- if a new block with data is available, we can directly start a new readout sequence
			readstate	<= STARTREADOUT;
			trigtoggle	<= not trigtoggle;
		end if;

-- hitbus events		
	when HBHEADER =>
		readhbfifo 			<= '0';				
		data_out				<= '0' & hbcounter(30 downto 0);
		out_ena				<= '1';
		hbcounter 			<= hbcounter + '1';
		readstate			<= HBDATA_MSB;

	when HBDATA_MSB =>
		data_out			<= fromhbfifo(REG64_TOP_RANGE);
		out_ena			<= '1';
		if(hbfifoempty = '1')then
			readhbfifo 	<= '0';
		else
			readhbfifo 	<= '1';					
		end if;
		readstate 		<= HBDATA_LSB;
		
	when HBDATA_LSB =>
		data_out		<= fromhbfifo(REG64_BOTTOM_RANGE);
		out_ena			<= '1';
		readhbfifo 		<= '0';						
		if(hbfifoempty = '1')then
			readstate 	<= HBFOOTER;
		else
			readstate 	<= HBDATA_MSB;						
		end if;					
		
	when HBFOOTER =>
		-- End of triggers, go back to start
		data_out 		<= ENDOFHB;
		out_ena			<= '1';
		out_eoe			<= '1';
		readstate		<= IDLE;	
		if	waitforblock = '0' then		-- if a new block with data is available, we can directly start a new readout sequence
			readstate	<= STARTREADOUT;
			trigtoggle	<= not trigtoggle;
		end if;

	when others =>
		-- should actually never occur
		readstate <= IDLE;				
		
		
	end case;
	
end if;
end process;

-- output of hit counters
process(writeclk, reset_n)
	variable receivedhits_r:		errcounter_t;
	variable outoftimehits_r:		errcounter_t;
	variable intimehits_r:			errcounter_t;
	variable memwritehits_r:     	errcounter_t;
	variable overflowhits_r:     	errcounter_t;
begin
if(reset_n = '0') then
	received_hits		<= (others => '0');
	outoftime_hits		<= (others => '0');
	intime_hits			<= (others => '0');
	memwrite_hits		<= (others => '0');
	overflow_hits		<= (others => '0');
	receivedhits_r		:= (others => '0');
	outoftimehits_r	:= (others => '0');
	intimehits_r		:= (others => '0');
	memwritehits_r		:= (others => '0');
	overflowhits_r		:= (others => '0');
elsif(writeclk'event and writeclk = '1') then	
	receivedhits_r		:= (others => '0');
	outoftimehits_r	:= (others => '0');
	intimehits_r		:= (others => '0');
	memwritehits_r		:= (others => '0');
	overflowhits_r		:= (others => '0');
	for i in NINPUT-1 downto 0 loop
		receivedhits_r		:= receivedhits_r  + receivedhits(i);
		outoftimehits_r	:= outoftimehits_r + outoftimehits(i);
		intimehits_r		:= intimehits_r    + intimehits(i);
		memwritehits_r		:= memwritehits_r  + memwritehits(i);
		overflowhits_r		:= overflowhits_r	 + overflowhits(i);
	end loop;
	received_hits	<= receivedhits_r;
	outoftime_hits	<= outoftimehits_r;
	intime_hits		<= intimehits_r;
	memwrite_hits	<= memwritehits_r;
	overflow_hits	<= overflowhits_r;
end if;
end process;

process(readclk, reset_n)
begin
if(reset_n = '0') then
	sent_hits			<= (others => '0');
	break_counter		<= (others => '0');
elsif(readclk'event and readclk = '1') then	
	sent_hits		<= senthits;
	break_counter	<= break_counter_reg;
end if;
end process;

end rtl;
	
