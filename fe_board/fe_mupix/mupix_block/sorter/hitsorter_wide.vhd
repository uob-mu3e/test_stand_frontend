-- Sort hits by timestamp
-- New version for up to 45 input links and with memory for counter transmission
-- November 2019
-- niberger@uni-mainz.de


-- General idea: Write hits to a memory location according to their timestamp; 
-- one memory per chip, 16 (maybe 8?) slots in meory per chip and timestamp
-- After a fixed delay, the counters of how many hits there are get collected and are transferred to the
-- read side via another memory.
-- As the timestamps run at double the frequency of this entity, this part is duplicted for even and odd
-- timestamps, so they can be treated in parallel.

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;
use work.datapath_components.all;
use work.detectorfpga_constants.all;
use work.detectorfpga_types.all;

entity hitsorter_wide is 
	port (
		reset_n							: in std_logic;										-- async reset
		writeclk						: in std_logic;										-- clock for write/input side
		running							: in std_logic;
		currentts						: in slowts_t;										-- Upper 10 bits of the 11 bit ts
		hit_in							: in hit_array;
		hit_ena_in						: in std_logic_vector(NCHIPS-1 downto 0);			-- valid hit
		readclk							: in std_logic;										-- clock for read/output side
		data_out						: out reg32;										-- packaged data out
		out_ena							: out STD_LOGIC;									-- valid output data
		out_type						: out std_logic_vector(3 downto 0);				-- start/end of an output package, hits, end of run
		diagnostic_sel					: in std_logic_vector(5 downto 0);					-- control the multiplexer for diagnostic signals
		diagnostic_out					: out reg32											-- diganostic out (counters for hits at various stages)
		);
end hitsorter_wide;

architecture rtl of hitsorter_wide is

-- For run start/stop process
signal running_last:   std_logic;
signal running_read:   std_logic;
signal running_seq:	   std_logic;

signal tslow 	: slowts_t;
signal tshi  	: slowts_t;
signal tsread	: slowts_t;

signal runstartup : std_logic;
signal runshutdown: std_logic;

-- For hit writing process
signal hit_last1:	 hit_array;
signal hit_last2:	 hit_array;
signal hit_last3:	 hit_array;

signal hit_ena_last1: std_logic_vector(NCHIPS-1 downto 0);
signal hit_ena_last2: std_logic_vector(NCHIPS-1 downto 0);
signal hit_ena_last3: std_logic_vector(NCHIPS-1 downto 0);

signal tshit : ts_array;
signal slowtshit : slowts_array;

signal sametsafternext: chip_bits_t;
signal sametsnext: chip_bits_t;

signal dcountertemp	: doublecounter_chiparray;
signal dcountertemp2: doublecounter_chiparray;

-- Actual sorter memory
signal tomem : nots_hit_array;
signal frommem : nots_hit_array;
signal memwren : std_logic_vector(NCHIPS-1 downto 0);
signal waddr	: addr_array;
signal raddr	: addr_array;

-- Counter memory
signal tocmem 	: alldoublecounter_array;
signal tocmem_hitwriter 	: alldoublecounter_array;
signal fromcmem	: alldoublecounter_array;
signal fromcmem_hitreader : doublecounter_chiparray;
signal cmemreadaddr : allcounteraddr_array;
signal cmemwriteaddr : allcounteraddr_array;
signal cmemreadaddr_hitwriter : allcounteraddr_array;
signal cmemwriteaddr_hitwriter : allcounteraddr_array;
signal cmemreadaddr_hitreader : counteraddr_t;
signal cmemwren		: allcounterwren_array;
signal cmemwren_hitwriter	: allcounterwren_array;

-- Fifo for counters to sequencer
signal reset : std_logic;
signal tofifo_counters : std_logic_vector(253 downto 0);
signal fromfifo_counters : std_logic_vector(253 downto 0);
signal read_counterfifo: std_logic;
signal write_counterfifo: std_logic;
signal counterfifo_almostfull: std_logic;
signal counterfifo_empty: std_logic;

signal block_nonempty_accumulate : std_logic;
signal block_empty : std_logic;
signal block_empty_del1 : std_logic;
signal block_empty_del2 : std_logic;

signal stopwrite : std_logic;
signal stopwrite_del1 : std_logic;
signal stopwrite_del2 : std_logic;

signal blockchange : std_logic;
signal blockchange_del1 : std_logic;
signal blockchange_del2 : std_logic;

constant counter2chipszero : counter2_chips := (others => '0');

signal even_nnonempty: std_logic_vector(3 downto 0);	
signal even_nechips	 : chip_bits_t;
signal even_nechips2 : chip_bits_t;
signal even_countchips: counter_chips;
signal even_countchips_m1: counter2_chips;
signal even_countchips_m2: counter2_chips;
signal haseven: std_logic;
signal even_overflow: std_logic;
signal even_overflow_del1: std_logic;
signal even_overflow_del2: std_logic;

signal odd_nnonempty: std_logic_vector(3 downto 0);	
signal odd_nechips	 : chip_bits_t;
signal odd_nechips2 : chip_bits_t;
signal odd_countchips: counter_chips;
signal odd_countchips_m1: counter2_chips;
signal odd_countchips_m2: counter2_chips;
signal hasodd: std_logic;
signal odd_overflow: std_logic;
signal odd_overflow_del1: std_logic;
signal odd_overflow_del2: std_logic;

signal credits: integer range -128 to 127;
signal credittemp : integer range -256 to 255;
signal hitcounter_sum_m3_even : hitcounter_sum3_type;
signal hitcounter_sum_m3_odd : hitcounter_sum3_type;
signal hitcounter_sum_even : integer;
signal hitcounter_sum_odd : integer;
signal hitcounter_sum : integer;

signal readcommand: 	command_t;
signal readcommand_last1: command_t;
signal readcommand_last2: command_t;
signal readcommand_last3: command_t;
signal readcommand_last4: command_t;
signal readcommand_ena:	std_logic;
signal readcommand_ena_last1:	std_logic;
signal readcommand_ena_last2:	std_logic;
signal readcommand_ena_last3:	std_logic;
signal readcommand_ena_last4:	std_logic;

signal outoverflow:	std_logic_vector(15 downto 0);
signal overflow_last1:	std_logic_vector(15 downto 0);
signal overflow_last2:	std_logic_vector(15 downto 0);
signal overflow_last3:	std_logic_vector(15 downto 0);
signal overflow_last4:	std_logic_vector(15 downto 0);

signal memmultiplex: nots_t;
signal tscounter: std_logic_vector(46 downto 0); --47 bit, LSB would run at double frequency, but not needed

-- diagnostics
signal noutoftime : reg_array;
signal noverflow  : reg_array;
signal nintime	  : reg_array;
signal nout		  : reg32;

constant TSONE : slowts_t := "0000000001";
constant TSZERO : slowts_t := "0000000000";
constant DELAY : slowts_t := "0110000000";
constant WINDOWSIZE : slowts_t := "1100000000";

begin


-- Generate timestamps that define windows for writing, reading and clearing
-- Including run start and stop logic
process(reset_n, writeclk)
begin
if(reset_n = '0') then
	running_last 	<= '0';
	running_read	<= '0';
	running_seq		<= '0';
	
	runstartup		<= '0';
	runshutdown		<= '0';
	
	tslow 		<= TSZERO;
	tshi		<= TSZERO;
	tsread		<= TSZERO;
elsif (writeclk'event and writeclk = '1') then

	running_last	<= running;

	if(running = '0') then
		runstartup		<= '0';
		runshutdown		<= '0';
	end if;
	
	if(running = '1' and running_last = '0') then
		runstartup <= '1';
	end if;
	
	if(running = '0' and running_last = '1') then
		runshutdown <= '1';
	end if;
	
	if(running = '1' and runstartup = '1') then
		tslow <= TSONE;
		tshi  <= WINDOWSIZE;
		if(currentts = DELAY) then
			runstartup <= '0';
		end if;
	elsif(running = '1' and running_last = '1' and runshutdown = '0') then
		tslow <= tslow + '1';
		tshi  <= tshi  + '1';
		running_read	<= '1';
		running_seq		<= '1';
	else
		tshi  <= tshi  + '1';
		tslow <= tslow + '1';
		if(tshi = TSZERO) then
			tshi	<= TSZERO;
			if(tslow = TSZERO) then
				tslow <= tszero;
				runshutdown <= '0';
				running_read <= '0';
				running_seq  <= '0';
			end if;
		end if;
	end if;
	
	tsread	  <= tslow - "11";
end if;
end process;


-- Memory for the actual sorting
genmem: for i in NCHIPS-1 downto 0 generate
	hsmem:hitsortermem
	PORT MAP
	(
		data				=> tomem(i),
		rdaddress			=> raddr(i),
		rdclock				=> writeclk,
		wraddress			=> waddr(i),
		wrclock				=> writeclk,
		wren				=> memwren(i),
		q					=> frommem(i)
	);

	-- In order to have enough ports also for clearing, we divide the memories for the counters
	-- into NMEMS memories, one address holds an even and an odd TS 
	gencmem: for k in NMEMS-1 downto 0 generate
		cmem:countermemory
			PORT MAP
			(
				clock				=> writeclk,
				data				=> tocmem(i)(k),
				rdaddress			=> cmemreadaddr(i)(k),
				wraddress			=> cmemwriteaddr(i)(k),
				wren				=> cmemwren(i)(k),
				q					=> fromcmem(i)(k)
			);
	
		tocmem(i)(k)			<= (others => '0') when k = conv_integer(tsread(SLOWTSCOUNTERMEMSELRANGE))
									else tocmem_hitwriter(i)(k);
		cmemreadaddr(i)(k)		<= 	cmemreadaddr_hitreader when 	k = conv_integer(tsread(SLOWTSCOUNTERMEMSELRANGE))
									else cmemreadaddr_hitwriter(i)(k);
		cmemwriteaddr(i)(k)		<= 	cmemreadaddr_hitreader when 	k = conv_integer(tsread(SLOWTSCOUNTERMEMSELRANGE))
									else cmemwriteaddr_hitwriter(i)(k);
		cmemwren(i)(k)			<= 	'1' when 	k = conv_integer(tsread(SLOWTSCOUNTERMEMSELRANGE))
									else cmemwren_hitwriter(i)(k);
	end generate gencmem;
	
	fromcmem_hitreader(i)	<= fromcmem(i)(conv_integer(tsread(SLOWTSCOUNTERMEMSELRANGE)));
	
	-- Write side: Put hits into memory at the right place and count them
	process(reset_n, writeclk)
		variable counterfrommem : doublecounter_t;
	begin
	if (reset_n = '0') then
		memwren(i) <= '0';		
		noutoftime(i) 	<= (others => '0');
		nintime(i)		<= (others => '0');
		noverflow(i)	<= (others => '0');
		
		for k in NMEMS-1 downto 0 loop
			cmemwren_hitwriter(i)(k) <= '0'; 	
			cmemreadaddr_hitwriter(i)(k)	<= (others => '0');
			cmemwriteaddr_hitwriter(i)(k)	<= (others => '0');
		end loop;
		
		hit_ena_last1(i)	<= '0';
		hit_ena_last2(i)	<= '0';
		hit_ena_last3(i)	<= '0';
		
		hit_last1(i)	<= (others => '0');
		hit_last2(i)	<= (others => '0');
		hit_last3(i)	<= (others => '0');
		
		sametsnext(i)		<= '0';
		sametsafternext(i)	<= '0';
		
	elsif (writeclk'event and writeclk = '1') then
		memwren(i) <= '0';
		
		tshit(i) 		<= hit_last1(i)(TSRANGE);
		slowtshit(i) 	<= hit_last1(i)(SLOWTSRANGE);
		
		hit_last1(i) <= hit_in(i);
		hit_last2(i) <= hit_last1(i);
		hit_last3(i) <= hit_last2(i);
		
		hit_ena_last1(i)	<= hit_ena_in(i);
		hit_ena_last2(i)	<= hit_ena_last1(i);
		hit_ena_last3(i)	<= hit_ena_last2(i);
		
		tomem(i) <= hit_last2(i)(NOTSRANGE);
		
		for k in NMEMS-1 downto 0 loop
			cmemreadaddr_hitwriter(i)(k)	<= hit_in(i)(COUNTERMEMADDRRANGE);
			cmemwriteaddr_hitwriter(i)(k)  <= hit_last2(i)(COUNTERMEMADDRRANGE);
		end loop;
		
		counterfrommem := fromcmem(i)(conv_integer(hit_last1(i)(COUNTERMEMSELRANGE)));
		
		for k in NMEMS-1 downto 0 loop
			cmemwren_hitwriter(i)(k) <= '0'; 		
		end loop;
	
		-- Reading from the memory, incrementing the counter and storing it again takes three
		-- cycles, so we cannot rely on what was written to the memory for incrementing and have to deal
		-- with this out-of-memory
		if(hit_ena_last2(i) = '1' and hit_last1(i)(SLOWTSRANGE) = hit_last2(i)(SLOWTSRANGE)) then
			sametsnext(i)	<= '1';
			sametsafternext(i)	<= '0';
		elsif(hit_ena_last3(i) = '1' and hit_last1(i)(SLOWTSRANGE) = hit_last3(i)(SLOWTSRANGE)) then
			sametsnext(i)	<= '0';
			sametsafternext(i)	<= '1';
		else
			sametsnext(i) <= '0';
			sametsafternext(i)	<= '0';
		end if;
		
		dcountertemp2(i) <= dcountertemp(i);
	
		if((running = '1' or runshutdown = '1') and hit_ena_last2(i) ='1') then -- Hit coming in during run
			if(((tshi > tslow) and (slowtshit(i) > tslow and slowtshit(i) < tshi)) or
				((tslow > tshi) and (slowtshit(i) > tslow or slowtshit(i) < tshi))) then
				-- Hit TS in the range we can accept
				if(hit_last2(i)(0) = '0') then -- even TS
					if(sametsnext(i) = '0' and sametsafternext(i) = '0') then -- not the same memory location as the last hit
						waddr(i) 	<= tshit(i) & counterfrommem(3 downto 0);
						if(counterfrommem(3 downto 0) /= "1111") then -- no overflow yet
							memwren(i)	<= '1';
							nintime(i)	<= nintime(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= counterfrommem(9 downto 5) & '0' & counterfrommem(3 downto 0) + '1';
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= counterfrommem(9 downto 5) & '0' & counterfrommem(3 downto 0) + '1';
						else -- overflow, mark this
							noverflow(i)	<= noverflow(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= counterfrommem(9 downto 5) & '1' & "1111";
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= counterfrommem(9 downto 5) & '1' & "1111";
						end if;
					elsif(sametsnext(i) = '1') then -- same memory location in last cycle
						waddr(i) 	<= tshit(i) & dcountertemp(i)(3 downto 0);
						if(dcountertemp(i)(3 downto 0) /= "1111") then -- no overflow yet
							nintime(i)	<= nintime(i) + '1';
							memwren(i)	<= '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= dcountertemp(i)(9 downto 5) & '0' & dcountertemp(i)(3 downto 0) + '1';
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= dcountertemp(i)(9 downto 5) & '0' & dcountertemp(i)(3 downto 0) + '1';
						else -- overflow, mark this
							noverflow(i)	<= noverflow(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= dcountertemp(i)(9 downto 5) & '1' & "1111";
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= dcountertemp(i)(9 downto 5) & '1' & "1111";
						end if;
					else -- same memory location two cycles ago
						waddr(i) 	<= tshit(i) & dcountertemp2(i)(3 downto 0);
						if(dcountertemp2(i)(3 downto 0) /= "1111") then -- no overflow yet
							nintime(i)	<= nintime(i) + '1';
							memwren(i)	<= '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= dcountertemp2(i)(9 downto 5) & '0' & dcountertemp2(i)(3 downto 0) + '1';
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= dcountertemp2(i)(9 downto 5) & '0' & dcountertemp2(i)(3 downto 0) + '1';
						else -- overflow, mark this
							noverflow(i)	<= noverflow(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= dcountertemp2(i)(9 downto 5) & '1' & "1111";
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= dcountertemp2(i)(9 downto 5) & '1' & "1111";
						end if;
					end if; -- same/ not same memory location
				else -- odd TS	
					if(sametsnext(i) = '0' and sametsafternext(i) = '0') then -- not the same memory location as the last hit
						waddr(i) 	<= tshit(i) & counterfrommem(8 downto 5);
						if(counterfrommem(8 downto 5) /= "1111") then -- no overflow yet
							memwren(i)	<= '1';
							nintime(i)	<= nintime(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= '0' & counterfrommem(8 downto 5) + '1' & counterfrommem(4 downto 0);
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= '0' & counterfrommem(8 downto 5) + '1' & counterfrommem(4 downto 0);
						else -- overflow, mark this
							noverflow(i)	<= noverflow(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= '1' & "1111"  & counterfrommem(4 downto 0);
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= '1' & "1111"  & counterfrommem(4 downto 0);
						end if;
					elsif(sametsnext(i) = '1') then -- same memory location in last cycle 
						waddr(i) 	<= tshit(i) & dcountertemp(i)(8 downto 5);
						if(dcountertemp(i)(8 downto 5) /= "1111") then -- no overflow yet
							memwren(i)	<= '1';
							nintime(i)	<= nintime(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= '0' & dcountertemp(i)(8 downto 5)  + '1' & dcountertemp(i)(4 downto 0);
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= '0' & dcountertemp(i)(8 downto 5)  + '1' & dcountertemp(i)(4 downto 0);
						else -- overflow, mark this
							noverflow(i)	<= noverflow(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= '1' & "1111" & dcountertemp(i)(4 downto 0);
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<=  '1' & "1111" & dcountertemp(i)(4 downto 0);
						end if; -- overflow
					else
						waddr(i) 	<= tshit(i) & dcountertemp2(i)(8 downto 5);
						if(dcountertemp2(i)(8 downto 5) /= "1111") then -- no overflow yet
							memwren(i)	<= '1';
							nintime(i)	<= nintime(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= '0' & dcountertemp2(i)(8 downto 5)  + '1' & dcountertemp2(i)(4 downto 0);
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<= '0' & dcountertemp2(i)(8 downto 5)  + '1' & dcountertemp2(i)(4 downto 0);
						else -- overflow, mark this
							noverflow(i)	<= noverflow(i) + '1';
							for k in NMEMS-1 downto 0 loop
								tocmem_hitwriter(i)(k)	<= '1' & "1111" & dcountertemp2(i)(4 downto 0);
							end loop;
							cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
							dcountertemp(i)	<=  '1' & "1111" & dcountertemp2(i)(4 downto 0);
						end if; -- overflow
					end if; -- same/ not same memory location
				end if; -- even odd TS
			else -- in/out of time
				-- we have an out of time hit: some diagnosis
				noutoftime(i) <= noutoftime(i) + '1';
			end if;
		end if; -- hit coming in during run;
	
	end if; -- clk event
	end process;
	
end generate genmem;	

reset <= not reset_n;

-- FIFO for passing counters to the sequencer
    cfifo: entity work.ip_scfifo
    generic map(
        ADDR_WIDTH      => 7,
        DATA_WIDTH      => 254,
        SHOWAHEAD       => "ON",
        DEVICE          => "ARRIA V",
        ALMOST_FULL     => 120--,
    )
    port map (
        clock           => writeclk,
        sclr            => reset,
        data            => tofifo_counters,
        wrreq           => write_counterfifo,
        almost_full     => counterfifo_almostfull,
        empty           => counterfifo_empty,
        q               => fromfifo_counters,
        rdreq           => read_counterfifo--,
    );

-- collect data for transmission to read side
-- read one line in the countermemories per cycle, condense counters and push to fifo if nonempty
process(reset_n, writeclk)
	variable even_ne : std_logic;
	variable even_ov : std_logic;
	variable even_nonemptycount : std_logic_vector(3 downto 0);	
	variable even_nfilled : integer;
	
	variable odd_ne : std_logic;
	variable odd_ov : std_logic;
	variable odd_nonemptycount : std_logic_vector(3 downto 0);	
	variable odd_nfilled : integer;
	
	variable countersum_even_temp : integer;
	variable countersum_odd_temp : integer;
	
	variable creditchange : integer;
	
begin
if (reset_n = '0') then
	cmemreadaddr_hitreader	<= (others => '0');
	write_counterfifo <= '0';
	block_nonempty_accumulate <= '0';
	block_empty	<= '0';
	block_empty_del1	<= '0';
	block_empty_del2	<= '0';
	
	stopwrite <= '0';
	stopwrite_del1 <= '0';	
	stopwrite_del2 <= '0';
	
	blockchange <= '0';
	blockchange_del1 <= '0';
	blockchange_del2 <= '0';
	
	credits <= 127;
	credittemp <= 127;
	for i in NCHIPS/3-1 downto 0 loop
		hitcounter_sum_m3_even(i) <= 0;
		hitcounter_sum_m3_odd(i)  <= 0;
	end loop;
	hitcounter_sum_even <= 0;
	hitcounter_sum_odd  <= 0;
	hitcounter_sum 		<= 0;
	
elsif (writeclk'event and writeclk = '1') then
	write_counterfifo <= '0';
	
	even_countchips_m1	<= (others => '0');
	even_countchips_m2	<= (others => '0');
	odd_countchips_m1	<= (others => '0');
	odd_countchips_m2	<= (others => '0');

	if(running_read = '1')then
		cmemreadaddr_hitreader	<= tsread(SLOWCOUNTERMEMADDRRANGE)+'1';
		
		-- or nonempty, read counters
		even_ov	:= '0';
		odd_ov	:= '0';
		for i in NCHIPS-1 downto 0 loop
			even_nechips(i) 	<= or_reduce(fromcmem_hitreader(i)(3 downto 0));
			even_countchips(i)	<= fromcmem_hitreader(i)(3 downto 0);
			even_ov				:= even_ov or fromcmem_hitreader(i)(4);
			
			odd_nechips(i) 		<= or_reduce(fromcmem_hitreader(i)(8 downto 5));
			odd_countchips(i)	<= fromcmem_hitreader(i)(8 downto 5);
			odd_ov				:= odd_ov or fromcmem_hitreader(i)(9);
		end loop;
		
		even_overflow <= even_ov;
		odd_overflow  <= odd_ov;
	
		even_nonemptycount := (others => '0');
		odd_nonemptycount := (others => '0');
		even_ne := '0';
		odd_ne	:= '0';
		for i in NCHIPS-1 downto 0 loop
			even_ne := even_ne or even_nechips(i);
			even_nonemptycount := even_nonemptycount + even_nechips(i);
			
			odd_ne := odd_ne or odd_nechips(i);
			odd_nonemptycount := odd_nonemptycount + odd_nechips(i);
		end loop;
		even_nnonempty 	<= even_nonemptycount;
		
		odd_nnonempty 	<= odd_nonemptycount;
		
		block_empty <= '0';
		blockchange	<= '0';
		if((or_reduce(tsread(SLOWTSNONBLOCKRANGE))) = '0') then
			blockchange <= '1';
			block_nonempty_accumulate <= odd_ne or even_ne;
			if(block_nonempty_accumulate = '0') then
				block_empty <= '1';
			end if;
			if(counterfifo_almostfull = '1' or credits <= 0) then
				stopwrite <= '1';
			else
				stopwrite <= '0';
			end if;
		else
			block_nonempty_accumulate <= odd_ne or even_ne or block_nonempty_accumulate;
		end if;
		
		
		
		-- multiplexing of counters -- here we pack groups of three towards the LSB
		-- Even
		even_nechips2	<= (others => '0');
		for i in NCHIPS/3-1 downto 0 loop
			hitcounter_sum_m3_even(i) <= conv_integer(even_countchips(3*i))
										+ conv_integer(even_countchips(3*i+1))
										+ conv_integer(even_countchips(3*i+2));
			if(even_nechips(i*3) = '1')then
				even_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= even_countchips(3*i);
				even_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+0, H);
				even_nechips2(i*3) <= '1';
				if(even_nechips(i*3+1) = '1')then
					even_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= even_countchips(3*i+1);
					even_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+1, H);
					even_nechips2(i*3+1) <= '1';
					if(even_nechips(i*3+2) = '1')then
						even_countchips_m1(H*2*3*i + 5*H-1 downto H*2*3*i + 4*H) <= even_countchips(3*i+2);
						even_countchips_m1(H*2*3*i + 6*H-1 downto H*2*3*i + 5*H) <= conv_std_logic_vector(3*i+2, H);
						even_nechips2(i*3+2) <= '1';
					end if;
				elsif(even_nechips(i*3+2) = '1')then
					even_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= even_countchips(3*i+2);
					even_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+2, H);
					even_nechips2(i*3+1) <= '1';
				end if;
				
			elsif(even_nechips(i*3+1) = '1')then
				even_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= even_countchips(3*i+1);
				even_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+1, H);
				even_nechips2(i*3) <= '1';
				if(even_nechips(i*3+2) = '1')then
					even_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= even_countchips(3*i+2);
					even_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+2, H);
					even_nechips2(i*3+1) <= '1';
				end if;
			elsif(even_nechips(i*3+2) = '1')then
				even_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= even_countchips(3*i+2);
				even_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+2, H);
				even_nechips2(i*3) <= '1';
			end if;
		end loop;
		
		-- Odd
		odd_nechips2	<= (others => '0');
		for i in NCHIPS/3-1 downto 0 loop
			hitcounter_sum_m3_odd(i) <= conv_integer(odd_countchips(3*i))
										+ conv_integer(odd_countchips(3*i+1))
										+ conv_integer(odd_countchips(3*i+2));

			if(odd_nechips(i*3) = '1')then
				odd_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= odd_countchips(3*i);
				odd_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+0, H);
				odd_nechips2(i*3) <= '1';
				if(odd_nechips(i*3+1) = '1')then
					odd_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= odd_countchips(3*i+1);
					odd_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+1, H);
					odd_nechips2(i*3+1) <= '1';
					if(odd_nechips(i*3+2) = '1')then
						odd_countchips_m1(H*2*3*i + 5*H-1 downto H*2*3*i + 4*H) <= odd_countchips(3*i+2);
						odd_countchips_m1(H*2*3*i + 6*H-1 downto H*2*3*i + 5*H) <= conv_std_logic_vector(3*i+2, H);
						odd_nechips2(i*3+2) <= '1';
					end if;
				elsif(odd_nechips(i*3+2) = '1')then
					odd_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= odd_countchips(3*i+2);
					odd_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+2, H);
					odd_nechips2(i*3+1) <= '1';
				end if;
				
			elsif(odd_nechips(i*3+1) = '1')then
				odd_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= odd_countchips(3*i+1);
				odd_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+1, H);
				odd_nechips2(i*3) <= '1';
				if(odd_nechips(i*3+2) = '1')then
					odd_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= odd_countchips(3*i+2);
					odd_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+2, H);
					odd_nechips2(i*3+1) <= '1';
				end if;
			elsif(odd_nechips(i*3+2) = '1')then
				odd_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= odd_countchips(3*i+2);
				odd_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+2, H);
				odd_nechips2(i*3) <= '1';
			end if;
		end loop;
		
		even_overflow_del1 	<= even_overflow;
		odd_overflow_del1 	<= odd_overflow;
		block_empty_del1	<= block_empty;
		stopwrite_del1		<= stopwrite;
		blockchange_del1	<= blockchange;
		
		-- multiplexing of counters, step 2
		--Even
		haseven <= or_reduce(even_nechips2);
		even_nfilled := 0;
		countersum_even_temp := 0;
		for i in  0 to NCHIPS/3-1 loop
			countersum_even_temp := countersum_even_temp + hitcounter_sum_m3_even(i);
			even_countchips_m2(2*H*even_nfilled + 2*3*H-1 downto 2*H*even_nfilled) 
			<= even_countchips_m1(2*3*i*H + 2*3*H-1 downto 2*3*i*H);
			if(even_nechips2(i*3+2 downto i*3) = "001") then
				even_nfilled := even_nfilled + 1;
			elsif(even_nechips2(i*3+2 downto i*3) = "011") then
				even_nfilled := even_nfilled + 2;
			elsif(even_nechips2(i*3+2 downto i*3) = "111") then
				even_nfilled := even_nfilled + 3;
			end if; 
		end loop;
		hitcounter_sum_even <= countersum_even_temp;
		
		-- Odd
		hasodd <= or_reduce(odd_nechips2);
		odd_nfilled := 0;
		countersum_odd_temp := 0;
		for i in  0 to NCHIPS/3-1 loop
			countersum_odd_temp := countersum_even_temp + hitcounter_sum_m3_even(i);
			odd_countchips_m2(2*H*odd_nfilled + 2*3*H-1 downto 2*H*odd_nfilled) 
			<= odd_countchips_m1(2*3*i*H + 2*3*H-1 downto 2*3*i*H);
			if(odd_nechips2(i*3+2 downto i*3) = "001") then
				odd_nfilled := odd_nfilled + 1;
			elsif(odd_nechips2(i*3+2 downto i*3) = "011") then
				odd_nfilled := odd_nfilled + 2;
			elsif(odd_nechips2(i*3+2 downto i*3) = "111") then
				odd_nfilled := odd_nfilled + 3;
			end if; 
		end loop;
		hitcounter_sum_odd <= countersum_odd_temp;
		
		even_overflow_del2 	<= even_overflow_del1;
		odd_overflow_del2 	<= odd_overflow_del1;
		block_empty_del2	<= block_empty_del1;
		stopwrite_del2		<= stopwrite_del1;
		blockchange_del2	<= blockchange_del1;
		
		tofifo_counters <= X"000000000000" & tsread - "100" & hasodd & odd_overflow_del2 & odd_countchips_m2 & haseven & even_overflow_del2 & even_countchips_m2;
	
		creditchange := 1;
		if(stopwrite_del2 = '0' and (hasodd = '1' or haseven = '1' or block_empty_del2 = '1')) then
			write_counterfifo <= '1';
			if(hitcounter_sum_even < 48) then -- limit number of hits per ts
				creditchange := creditchange - hitcounter_sum_even;
			else
				tofifo_counters(HASEVENBIT) 		<= '0';
				tofifo_counters(EVENOVERFLOWBIT)	<= '1';
				tofifo_counters(EVENCOUNTERRANGE)	<= counter2chipszero;
			end if;
			
			if(hitcounter_sum_odd < 48) then -- limit number of hits per ts
				creditchange := creditchange - hitcounter_sum_odd;
			else
				tofifo_counters(HASODDBIT) 		<= '0';
				tofifo_counters(ODDOVERFLOWBIT)	<= '1';
				tofifo_counters(ODDCOUNTERRANGE)<= counter2chipszero;
			end if;
			
			if(blockchange_del2 = '1') then
				creditchange := creditchange  -1;
			end if;
		elsif(stopwrite_del2 ='1' and block_empty_del2 = '1') then -- we were overfull but just got an empty block
			write_counterfifo <= '1';
			creditchange := creditchange  -1;
		elsif(stopwrite_del2 ='1' and blockchange_del2 = '1') then -- we were overfull and have suppressed hits
			write_counterfifo <= '1';
			tofifo_counters <= X"000000000000" & tsread - "100" & "0" & "1" & odd_countchips_m2 & "0" & "1" & even_countchips_m2;
			creditchange := creditchange  -1;
		end if;
		credittemp <= credittemp + creditchange;
		if(credittemp > 127) then
			credits <= 127;
			credittemp <= 127;
		elsif(credittemp < -128) then
			credits <= -128;
			credittemp <= 128;
		else
			credits <= credittemp;
		end if;
	end if;
end if;
end process;



-- Here we generate the sequence of read commands etc.
seq:sequencer 
	port map(
		reset_n							=> reset_n,
		clk								=> writeclk,
		from_fifo						=> fromfifo_counters,
		fifo_empty						=> counterfifo_empty,
		read_fifo						=> read_counterfifo,
		outcommand						=> readcommand,
		command_enable					=> readcommand_ena,
		outoverflow						=> outoverflow
		);
-- The ouput command has the TS in the LSBs, followed by four bits hit address
-- four bits channel/chip ID and the MSB inciating command (1) or hit (0)					
-- And the reading (use writeclk for the moment, FIFO comes after)
process(writeclk, reset_n)
begin
if(reset_n = '0') then
	data_out						<= (others => '0');
	out_ena							<= '0';
	out_type						<= (others => '0');
	readcommand_ena_last1			<= '0';
	readcommand_ena_last2			<= '0';
	readcommand_ena_last3			<= '0';	
	readcommand_ena_last4			<= '0';		
	tscounter						<= (others => '0');
	nout							<= (others => '0');
elsif(writeclk'event and writeclk = '1') then
	out_ena							<= '0';
	for i in NCHIPS-1 downto 0 loop
		raddr(i)							<= 	readcommand(TSRANGE) & --MSBs: Timestamp 
												readcommand(COMMANDBITS-6 downto TIMESTAMPSIZE); -- LSBs: hit address in TS
	end loop;

	readcommand_last1				<= readcommand;
	readcommand_last2				<= readcommand_last1;
	readcommand_last3				<= readcommand_last2;
	readcommand_last4				<= readcommand_last3;	
	
	readcommand_ena_last1			<= readcommand_ena;
	readcommand_ena_last2			<= readcommand_ena_last1;	
	readcommand_ena_last3			<= readcommand_ena_last2;
	readcommand_ena_last4			<= readcommand_ena_last3;
	
	overflow_last1					<= outoverflow;
	overflow_last2					<= overflow_last1;
	overflow_last3					<= overflow_last2;
	overflow_last4					<= overflow_last3;	

	out_ena							<= readcommand_ena_last4;
	
	if(conv_integer(readcommand_last3(COMMANDBITS-6 downto TIMESTAMPSIZE)) < NCHIPS) then
		memmultiplex						<= frommem(conv_integer(readcommand_last3(COMMANDBITS-2 downto TIMESTAMPSIZE+4)));
	end if;
	
	if(running_seq = '1') then
		tscounter <= tscounter + '1';
	end if;

	case readcommand_last4(COMMANDBITS-1 downto COMMANDBITS-4) is
	when COMMAND_HEADER1(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= tscounter(46 downto 15);
		out_type		<= "0010";
	when COMMAND_HEADER2(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= tscounter(14 downto 0) & '0' & X"0000";
		out_type		<= "0000";
	when COMMAND_SUBHEADER(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= "000" & readcommand_last4(TIMESTAMPSIZE-1) & "111111" & readcommand_last4(TIMESTAMPSIZE-2 downto 4) & overflow_last4;
		out_type		<= "0000";
	when COMMAND_FOOTER(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out 		<= (others => '0');
		out_type		<= "0011";
	when others =>
		data_out		<= readcommand_last4(3 downto 0) & "000" & readcommand_last4(COMMANDBITS-6 downto TIMESTAMPSIZE) & memmultiplex & '0';
		out_type		<= "0000";
		if(readcommand_ena_last4 = '1') then
			nout <= nout + '1';
		end if;
	end case;
end if;
end process;

sdm: sorter_diagnostic_mux
	PORT MAP
 	(
		aclr		=> reset,
		clock		=> writeclk,
		data0x		=> nintime(0),
		data1x		=> nintime(1),
		data2x		=> nintime(2),
		data3x		=> nintime(3),
		data4x		=> nintime(4),
		data5x		=> nintime(5),
		data6x		=> nintime(6),
		data7x		=> nintime(7),
		data8x		=> nintime(8),
		data9x		=> nintime(9),
		data10x		=> nintime(10),
		data11x		=> nintime(11),
		data12x		=> (others => '0'),--nintime(12),
		data13x		=> (others => '0'),--nintime(13),
		data14x		=> (others => '0'),--nintime(14),
		data15x		=> noutoftime(0),
		data16x		=> noutoftime(1),
		data17x		=> noutoftime(2),
		data18x		=> noutoftime(3),
		data19x		=> noutoftime(4),
		data20x		=> noutoftime(5),
		data21x		=> noutoftime(6),
		data22x		=> noutoftime(7),
		data23x		=> noutoftime(8),
		data24x		=> noutoftime(9),
		data25x		=> noutoftime(10),
		data26x		=> noutoftime(11),
		data27x		=> (others => '0'),--noutoftime(12),
		data28x		=> (others => '0'),--noutoftime(13),
		data29x		=> (others => '0'),--noutoftime(14),
		data30x		=> noverflow(0),
		data31x		=> noverflow(1),
		data32x		=> noverflow(2),
		data33x		=> noverflow(3),
		data34x		=> noverflow(4),
		data35x		=> noverflow(5),
		data36x		=> noverflow(6),
		data37x		=> noverflow(7),
		data38x		=> noverflow(8),
		data39x		=> noverflow(9),
		data40x		=> noverflow(10),
		data41x		=> noverflow(11),
		data42x		=> (others => '0'),--noverflow(12),
		data43x		=> (others => '0'),--noverflow(13),
		data44x		=> (others => '0'),--noverflow(14),
		data45x		=> (others => '0'),
		data46x		=> (others => '0'),
		data47x		=> (others => '0'),
		data48x		=> (others => '0'),
		data49x		=> (others => '0'),
		data50x		=> (others => '0'),
		data51x		=> (others => '0'),
		data52x		=> (others => '0'),
		data53x		=> (others => '0'),
		data54x		=> (others => '0'),
		data55x		=> (others => '0'),
		data56x		=> (others => '0'),
		data57x		=> (others => '0'),
		data58x		=> (others => '0'),
		data59x		=> (others => '0'),
		data60x		=> (others => '0'),
		data61x		=> (others => '0'),
		data62x		=> (others => '0'),
		data63x		=> nout,
		sel			=> diagnostic_sel,
		result		=> diagnostic_out
	);

end architecture RTL;