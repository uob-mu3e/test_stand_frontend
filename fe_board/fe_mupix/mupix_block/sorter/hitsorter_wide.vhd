-- Sort hits by timestamp
-- Version for spring 2021: Up to 36 input links, 125 MHz TS
-- February 2021, Niklaus Berger
-- niberger@uni-mainz.de


-- General idea: Write hits to a memory location according to their timestamp; 
-- one memory per chip, 16  slots in memory per chip and timestamp
-- After a fixed delay, the counters of how many hits there are get collected and are transferred to the
-- read side via another memory.

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;

use work.mupix.all;
use work.mudaq.all;

LIBRARY altera_mf;
USE altera_mf.all;


entity hitsorter_wide is 
    port (
        reset_n         : in  std_logic;                            -- async reset
        writeclk        : in  std_logic;                            -- clock for write/input side
        running         : in  std_logic;
        currentts       : in  ts_t;                                 -- 11 bit ts
        hit_in          : in  hit_array;
        hit_ena_in      : in  std_logic_vector(NCHIPS-1 downto 0);  -- valid hit
        readclk         : in  std_logic;                            -- clock for read/output side
        data_out        : out reg32;                                -- packaged data out
        out_ena         : out STD_LOGIC;                            -- valid output data
        out_type        : out std_logic_vector(3 downto 0);         -- start/end of an output package, hits, end of run		
        out_is_hit      : out std_logic;                            -- same as out_ena, but only hits, no trailer header etc.

        i_clk156        : in  std_logic;
        i_reset_n_regs  : in  std_logic;
        i_reg_add       : in  std_logic_vector(15 downto 0);
        i_reg_re        : in  std_logic;
        o_reg_rdata     : out std_logic_vector(31 downto 0);
        i_reg_we        : in  std_logic;
        i_reg_wdata     : in  std_logic_vector(31 downto 0)--;
    );
end hitsorter_wide;

architecture rtl of hitsorter_wide is

-- For run start/stop process
signal running_last:   std_logic;
signal running_read:   std_logic;
signal running_read_last:   std_logic;
signal running_read_last2:   std_logic;
signal running_seq:	   std_logic;

signal tslow 	: ts_t;
signal tshi  	: ts_t;
signal tsread	: ts_t;
signal tsreadmemdelay	: ts_t;

signal runstartup : std_logic;
signal runshutdown: std_logic;
signal runend	  : std_logic;

-- For hit writing process
signal hit_last1:	 hit_array;
signal hit_last2:	 hit_array;
signal hit_last3:	 hit_array;

signal hit_ena_last1: std_logic_vector(NCHIPS-1 downto 0);
signal hit_ena_last2: std_logic_vector(NCHIPS-1 downto 0);
signal hit_ena_last3: std_logic_vector(NCHIPS-1 downto 0);

signal tshit : ts_array;

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
signal addrcounterreset : counteraddr_t := (others => '0');
signal cmemwren		: allcounterwren_array;
signal cmemwren_hitwriter	: allcounterwren_array;

-- Fifo for counters to sequencer
signal reset : std_logic;
signal tofifo_counters : sorterfifodata_t;
signal fromfifo_counters : sorterfifodata_t;
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
signal stopwrite_del3 : std_logic;

signal blockchange : std_logic;
signal blockchange_del1 : std_logic;
signal blockchange_del2 : std_logic;

constant counter2chipszero : counter2_chips := (others => '0');

signal mem_nnonempty: std_logic_vector(3 downto 0);	
signal mem_nechips	 : chip_bits_t;
signal mem_nechips2 : chip_bits_t;
signal mem_countchips: counter_chips;
signal mem_countchips_m1: counter2_chips;
signal mem_countchips_m2: counter2_chips;
signal hashits: std_logic;
signal mem_overflow: std_logic;
signal mem_overflow_del1: std_logic;
signal mem_overflow_del2: std_logic;

signal credits: integer range -128 to 127;
signal credits32: std_logic_vector(31 downto 0);
signal credittemp : integer range -256 to 255;
signal hitcounter_sum_m3_mem : hitcounter_sum3_type;
signal hitcounter_sum_mem : integer;
signal hitcounter_sum : integer;
signal creditchange_reg : integer;

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

signal header_counter: std_logic_vector(15 downto 0);
signal subheader_counter: std_logic_vector(15 downto 0);

signal memmultiplex: nots_t;
signal tscounter: std_logic_vector(47 downto 0); --47 bit, LSB would run at double frequency, but not needed

-- end of run sequence on output side
signal terminate_output : std_logic;
signal terminated_output : std_logic;

signal debug_subheadercounter : std_logic_vector(15 downto 0);
signal debug_hitcounter 	  : std_logic_vector(15 downto 0);

-- diagnostics
signal noutoftime       : reg_array;
signal noverflow        : reg_array;
signal nintime          : reg_array;
signal nout             : reg32;
signal delay            : ts_t;

-- copy of diagnostics (timing)
signal noutoftime2: reg_array;
signal noverflow2 : reg_array;
signal nintime2   : reg_array;
signal nout2      : reg32;

constant TSONE : ts_t := "00000000001";
constant TSZERO : ts_t := "00000000000";
constant TSTHREE : ts_t := "00000000011";
--constant DELAY : ts_t := "01100000000";
constant WINDOWSIZE : ts_t := "10000000000";
constant READOFFSET : ts_t := "00100000011";


        COMPONENT scfifo
        GENERIC (
                add_ram_output_register         : STRING;
                almost_full_value               : NATURAL;
                intended_device_family          : STRING;
                lpm_numwords            : NATURAL;
                lpm_showahead           : STRING;
                lpm_type                : STRING;
                lpm_width               : NATURAL;
                lpm_widthu              : NATURAL;
                overflow_checking               : STRING;
                underflow_checking              : STRING;
                use_eab         : STRING
        );
        PORT (
                        aclr    : IN STD_LOGIC ;
                        clock   : IN STD_LOGIC ;
                        data    : IN sorterfifodata_t;
                        rdreq   : IN STD_LOGIC ;
                        sclr    : IN STD_LOGIC ;
                        wrreq   : IN STD_LOGIC ;
                        almost_full     : OUT STD_LOGIC ;
                        empty   : OUT STD_LOGIC ;
                        q       : OUT sorterfifodata_t
        );
        END COMPONENT;




begin


-- Generate timestamps that define windows for writing, reading and clearing
-- Including run start and stop logic
process(reset_n, writeclk)
begin
if(reset_n = '0') then
	running_last 	<= '0';
	running_read	<= '0';
	running_read_last	<= '0';
	running_read_last2	<= '0';
	running_seq		<= '0';
	
	runstartup		<= '0';
	runshutdown		<= '0';
	runend			<= '0';
	
	tslow 		<= TSZERO;
	tshi		<= TSZERO;
	tsread		<= TSZERO;
	tsreadmemdelay <= TSZERO;
elsif (writeclk'event and writeclk = '1') then

	tsread	  <= tslow - READOFFSET;
	tsreadmemdelay <= tsread;

	running_last	<= running;
	running_read_last  <= running_read;
	running_read_last2	<= running_read_last;

	if(running = '0') then
		runstartup		<= '0';
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
		if(currentts = WINDOWSIZE + delay - "11") then
			runstartup <= '0';
		end if;
	elsif(running = '1' and running_last = '1' and runshutdown = '0') then
		tslow <= tslow + '1';
		tshi  <= tshi  + '1';
		if(running_read = '0' and tslow >= READOFFSET) then
			running_read	<= '1';
			running_seq		<= '1';
		end if;
	elsif(runshutdown = '1') then-- shutdown sequence
		tshi  <= tshi  + '1';
		tslow <= tslow + '1';
		if(tshi = TSZERO) then
			tshi	<= TSZERO;
			if(tslow = TSZERO) then
				tslow <= tszero;
				tsread <= tsread + '1';
				if(tsread - "10100" = TSZERO) then
					running_read <= '0';
					runend		 <= '1';
					running_seq  <= '0';
				end if;				
			end if;
		end if;
	else
		tshi	<= TSZERO;	
		tslow 	<= TSZERO;
	end if;
end if;
end process;

-- we need to run through addresses for resetteing the memory
process(writeclk)
begin
if (writeclk'event and writeclk = '1') then
	addrcounterreset <= addrcounterreset + '1';
end if;
end process;


-- Memory for the actual sorting
genmem: for i in NCHIPS-1 downto 0 generate
	hsmem: entity work.hitsortermem_large
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
	-- into NMEMS memories, one address holds one TS 
	gencmem: for k in NMEMS-1 downto 0 generate
		cmem:entity work.countermemory
			PORT MAP
			(
				clock				=> writeclk,
				data				=> tocmem(i)(k),
				rdaddress			=> cmemreadaddr(i)(k),
				wraddress			=> cmemwriteaddr(i)(k),
				wren				=> cmemwren(i)(k),
				q					=> fromcmem(i)(k)
			);
	
		tocmem(i)(k)			<= (others => '0') when k = conv_integer(tsread(COUNTERMEMSELRANGE)) or reset_n = '0'
									else tocmem_hitwriter(i)(k);
		cmemreadaddr(i)(k)		<= 	cmemreadaddr_hitreader when 	k = conv_integer(tsread(COUNTERMEMSELRANGE))
									else cmemreadaddr_hitwriter(i)(k);
		cmemwriteaddr(i)(k)		<= 	cmemreadaddr_hitreader when 	k = conv_integer(tsread(COUNTERMEMSELRANGE)) or reset_n = '0'
									else cmemwriteaddr_hitwriter(i)(k);
		cmemwren(i)(k)			<= 	'1' when 	k = conv_integer(tsread(COUNTERMEMSELRANGE)) or reset_n = '0'
									else cmemwren_hitwriter(i)(k);
	end generate gencmem;
	
	fromcmem_hitreader(i)	<= fromcmem(i)(conv_integer(tsreadmemdelay(COUNTERMEMSELRANGE)));
	
	
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
			cmemwren_hitwriter(i)(k) 		<= '1'; 	
			cmemreadaddr_hitwriter(i)(k)	<= (others => '0');
			cmemwriteaddr_hitwriter(i)(k)	<= addrcounterreset;
		end loop;
		
		hit_ena_last1(i)	<= '0';
		hit_ena_last2(i)	<= '0';
		hit_ena_last3(i)	<= '0';
		
		hit_last1(i)	<= (others => '0');
		hit_last2(i)	<= (others => '0');
		hit_last3(i)	<= (others => '0');
		
		sametsnext(i)		<= '0';
		sametsafternext(i)	<= '0';

		for k in NMEMS-1 downto 0 loop
			tocmem_hitwriter(i)(k) <= (others => '0'); 		
		end loop;

		
	elsif (writeclk'event and writeclk = '1') then

		memwren(i) <= '0';
		
		tshit(i) 		<= hit_last1(i)(TSRANGE);
		
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
		
		counterfrommem := fromcmem(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE)));
		
		for k in NMEMS-1 downto 0 loop
			cmemwren_hitwriter(i)(k) <= '0'; 		
		end loop;
	
		-- Reading from the memory, incrementing the counter and storing it again takes three
		-- cycles, so we cannot rely on what was written to the memory for incrementing and have to deal
		-- with this out-of-memory
		if(hit_ena_last2(i) = '1' and hit_last1(i)(TSRANGE) = hit_last2(i)(TSRANGE)) then
			sametsnext(i)	<= '1';
			sametsafternext(i)	<= '0';
		elsif(hit_ena_last3(i) = '1' and hit_last1(i)(TSRANGE) = hit_last3(i)(TSRANGE)) then
			sametsnext(i)	<= '0';
			sametsafternext(i)	<= '1';
		else
			sametsnext(i) <= '0';
			sametsafternext(i)	<= '0';
		end if;
		
		dcountertemp2(i) <= dcountertemp(i);
	
		if((running = '1' or runshutdown = '1') and hit_ena_last2(i) ='1') then -- Hit coming in during run
			if(((tshi > tslow) and (tshit(i) >= tslow and tshit(i) < tshi)) or
				((tslow > tshi) and (tshit(i) >= tslow or tshit(i) < tshi))) then
				-- Hit TS in the range we can accept
				if(sametsnext(i) = '0' and sametsafternext(i) = '0') then -- not the same memory location as the last hit
					waddr(i) 	<= tshit(i) & counterfrommem(3 downto 0);
					if(counterfrommem(3 downto 0) /= "1111") then -- no overflow yet
						memwren(i)	<= '1';
						nintime(i)	<= nintime(i) + '1';
						for k in NMEMS-1 downto 0 loop
							tocmem_hitwriter(i)(k)	<=  '0' & counterfrommem(3 downto 0) + '1';
						end loop;
						cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
						dcountertemp(i)	<= '0' & counterfrommem(3 downto 0) + '1';
					else -- overflow, mark this
						noverflow(i)	<= noverflow(i) + '1';
						for k in NMEMS-1 downto 0 loop
							tocmem_hitwriter(i)(k)	<= '1' & "1111";
						end loop;
						cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
						dcountertemp(i)	<= '1' & "1111";
					end if;
				elsif(sametsnext(i) = '1') then -- same memory location in last cycle
					waddr(i) 	<= tshit(i) & dcountertemp(i)(3 downto 0);
					if(dcountertemp(i)(3 downto 0) /= "1111") then -- no overflow yet
						nintime(i)	<= nintime(i) + '1';
						memwren(i)	<= '1';
						for k in NMEMS-1 downto 0 loop
							tocmem_hitwriter(i)(k)	<= '0' & dcountertemp(i)(3 downto 0) + '1';
						end loop;
						cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
						dcountertemp(i)	<= '0' & dcountertemp(i)(3 downto 0) + '1';
					else -- overflow, mark this
						noverflow(i)	<= noverflow(i) + '1';
						for k in NMEMS-1 downto 0 loop
							tocmem_hitwriter(i)(k)	<= '1' & "1111";
						end loop;
						cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
						dcountertemp(i)	<= '1' & "1111";
					end if;
				else -- same memory location two cycles ago
					waddr(i) 	<= tshit(i) & dcountertemp2(i)(3 downto 0);
					if(dcountertemp2(i)(3 downto 0) /= "1111") then -- no overflow yet
						nintime(i)	<= nintime(i) + '1';
						memwren(i)	<= '1';
						for k in NMEMS-1 downto 0 loop
							tocmem_hitwriter(i)(k)	<= '0' & dcountertemp2(i)(3 downto 0) + '1';
						end loop;
						cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
						dcountertemp(i)	<= '0' & dcountertemp2(i)(3 downto 0) + '1';
					else -- overflow, mark this
						noverflow(i)	<= noverflow(i) + '1';
						for k in NMEMS-1 downto 0 loop
							tocmem_hitwriter(i)(k)	<= '1' & "1111";
						end loop;
						cmemwren_hitwriter(i)(conv_integer(hit_last2(i)(COUNTERMEMSELRANGE))) <= '1';
						dcountertemp(i)	<=  '1' & "1111";
					end if;
				end if; -- same/ not same memory location
			else -- in/out of time
				-- we have an out of time hit: some diagnosis
				noutoftime(i) <= noutoftime(i) + '1';
			end if;
		end if; -- hit coming in during run;
	
	end if; -- clk event
	end process;
	
end generate genmem;	

reset <= not reset_n;

scfifo_component : scfifo
        GENERIC MAP (
                add_ram_output_register => "ON",
                almost_full_value => 120,
                intended_device_family => "Arria V",
                lpm_numwords => 128,
                lpm_showahead => "OFF",
                lpm_type => "scfifo",
                lpm_width => SORTERFIFORANGE'left + 1,
                lpm_widthu => 7,
                overflow_checking => "ON",
                underflow_checking => "ON",
                use_eab => "ON"
        )
        PORT MAP (
                aclr => '0',
                clock => writeclk,
                data => tofifo_counters,
                rdreq => read_counterfifo,
                sclr => reset,
                wrreq => write_counterfifo,
                almost_full => counterfifo_almostfull,
                empty => counterfifo_empty,
                q => fromfifo_counters
        );



-- collect data for transmission to read side
-- read one line in the countermemories per cycle, condense counters and push to fifo if nonempty
process(reset_n, writeclk)
	variable mem_ne : std_logic;
	variable mem_ov : std_logic;
	variable mem_nonemptycount : std_logic_vector(3 downto 0);	
	variable mem_nfilled : integer;
	
	variable countersum_temp : integer;
	
	variable creditchange : integer range -2048 to 2047;
	
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
		hitcounter_sum_m3_mem(i) <= 0;
	end loop;
	hitcounter_sum_mem <= 0;
	hitcounter_sum 		<= 0;
	
elsif (writeclk'event and writeclk = '1') then
	write_counterfifo <= '0';
	
	mem_countchips_m1	<= (others => '0');
	mem_countchips_m2	<= (others => '0');

	
	if(running_read = '1')then
		cmemreadaddr_hitreader	<= tsread(COUNTERMEMADDRRANGE)+'1';
		
		-- or nonempty, read counters
		mem_ov	:= '0';
		for i in NCHIPS-1 downto 0 loop
			mem_nechips(i) 		<= or_reduce(fromcmem_hitreader(i)(3 downto 0));
			mem_countchips(i)	<= fromcmem_hitreader(i)(3 downto 0);
			mem_ov				:= mem_ov or fromcmem_hitreader(i)(4);
		end loop;
		
		mem_overflow <= mem_ov;
	
		
		mem_nonemptycount := (others => '0');
		mem_ne := '0';

		if(running_read_last2 = '1') then
			for i in NCHIPS-1 downto 0 loop
				mem_ne := mem_ne or mem_nechips(i);
				mem_nonemptycount := mem_nonemptycount + mem_nechips(i);
			end loop;
			mem_nnonempty 	<= mem_nonemptycount;
		end if;
		
		blockchange	<= '0';
		if((or_reduce(tsread(TSNONBLOCKRANGE))) = '0' and running_read_last = '1') then -- no block change at startup
			blockchange <= '1';
			if(counterfifo_almostfull = '1' or credits <= 0) then
				stopwrite <= '1';
			else
				stopwrite <= '0';
			end if;
		
		end if;
		
		block_empty_del2 <= '0';
		if(blockchange_del1 = '1') then
            block_nonempty_accumulate <= mem_ne;
            if(block_nonempty_accumulate = '0')then
                block_empty_del2 <= '1';
            end if;
		else
			block_nonempty_accumulate <= mem_ne or block_nonempty_accumulate;
		end if;
		
		
		-- multiplexing of counters -- here we pack groups of three towards the LSB
		-- Even
		mem_nechips2	<= (others => '0');
		for i in NCHIPS/3-1 downto 0 loop
			hitcounter_sum_m3_mem(i) <= conv_integer(mem_countchips(3*i))
										+ conv_integer(mem_countchips(3*i+1))
										+ conv_integer(mem_countchips(3*i+2));
			if(mem_nechips(i*3) = '1')then
				mem_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= mem_countchips(3*i);
				mem_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+0, H);
				mem_nechips2(i*3) <= '1';
				if(mem_nechips(i*3+1) = '1')then
					mem_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= mem_countchips(3*i+1);
					mem_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+1, H);
					mem_nechips2(i*3+1) <= '1';
					if(mem_nechips(i*3+2) = '1')then
						mem_countchips_m1(H*2*3*i + 5*H-1 downto H*2*3*i + 4*H) <= mem_countchips(3*i+2);
						mem_countchips_m1(H*2*3*i + 6*H-1 downto H*2*3*i + 5*H) <= conv_std_logic_vector(3*i+2, H);
						mem_nechips2(i*3+2) <= '1';
					end if;
				elsif(mem_nechips(i*3+2) = '1')then
					mem_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= mem_countchips(3*i+2);
					mem_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+2, H);
					mem_nechips2(i*3+1) <= '1';
				end if;
				
			elsif(mem_nechips(i*3+1) = '1')then
				mem_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= mem_countchips(3*i+1);
				mem_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+1, H);
				mem_nechips2(i*3) <= '1';
				if(mem_nechips(i*3+2) = '1')then
					mem_countchips_m1(H*2*3*i + 3*H-1 downto H*2*3*i + 2*H) <= mem_countchips(3*i+2);
					mem_countchips_m1(H*2*3*i + 4*H-1 downto H*2*3*i + 3*H) <= conv_std_logic_vector(3*i+2, H);
					mem_nechips2(i*3+1) <= '1';
				end if;
			elsif(mem_nechips(i*3+2) = '1')then
				mem_countchips_m1(H*2*3*i + H-1 downto H*2*3*i) <= mem_countchips(3*i+2);
				mem_countchips_m1(H*2*3*i + 2*H-1 downto H*2*3*i + H) <= conv_std_logic_vector(3*i+2, H);
				mem_nechips2(i*3) <= '1';
			end if;
		end loop;
		
		mem_overflow_del1 	<= mem_overflow;
		stopwrite_del1		<= stopwrite;
		blockchange_del1	<= blockchange;
		
		-- multiplexing of counters, step 2
		hashits <= or_reduce(mem_nechips2);
		mem_nfilled := 0;
		countersum_temp := 0;
		for i in  0 to NCHIPS/3-1 loop
			countersum_temp := countersum_temp + hitcounter_sum_m3_mem(i);
			mem_countchips_m2(2*H*mem_nfilled + 2*3*H-1 downto 2*H*mem_nfilled) 
			<= mem_countchips_m1(2*3*i*H + 2*3*H-1 downto 2*3*i*H);
			if(mem_nechips2(i*3+2 downto i*3) = "001") then
				mem_nfilled := mem_nfilled + 1;
			elsif(mem_nechips2(i*3+2 downto i*3) = "011") then
				mem_nfilled := mem_nfilled + 2;
			elsif(mem_nechips2(i*3+2 downto i*3) = "111") then
				mem_nfilled := mem_nfilled + 3;
			end if; 
		end loop;
		hitcounter_sum_mem <= countersum_temp;
		
		
		mem_overflow_del2 	<= mem_overflow_del1;
		stopwrite_del2		<= stopwrite_del1;
		blockchange_del2	<= blockchange_del1;

		-- one more delay cycle for stopwrite, as it supresses the NEXT block
		stopwrite_del3		<= stopwrite_del2;
		
		tofifo_counters <=  tsread - "100" & hashits & mem_overflow_del2 & mem_countchips_m2;
		--X"000000000000" &
		creditchange := 1;

		
		if(stopwrite_del3 = '0' and (hashits = '1' or block_empty_del2 = '1')) then
			write_counterfifo <= '1';
			if(hitcounter_sum_mem < 48) then -- limit number of hits per ts
				creditchange := creditchange - hitcounter_sum_mem;
			else
				tofifo_counters(HASMEMBIT) 		<= '1';
				tofifo_counters(MEMOVERFLOWBIT)	<= '1';
				tofifo_counters(MEMCOUNTERRANGE)	<= counter2chipszero;
				creditchange := creditchange - 1;
			end if;
						
			if(blockchange_del2 = '1') then
				creditchange := creditchange  -1;
			end if;

		elsif(stopwrite_del3 ='1' and blockchange_del2 = '1' and block_empty_del2 = '1') then -- we were overfull but just got an empty block
			write_counterfifo <= '1';
			creditchange := creditchange  -1;
		elsif(stopwrite_del3 ='1' and blockchange_del2 = '1') then -- we were overfull and have suppressed hits
			write_counterfifo <= '1';
			tofifo_counters <= tsread - "100" & "0" & "1" & counter2chipszero;
			creditchange := creditchange  -1;
		end if;
		credittemp <= credittemp + creditchange;
		creditchange_reg  <= creditchange;
		if(credittemp  + creditchange > 127) then
			credits <= 127;
			credittemp <= 127;
		elsif(credittemp +creditchange < -128) then
			credits <= -128;
			credittemp <= -128;
		else
			credits <= credittemp;
		end if;
	end if;
end if;
end process;



-- Here we generate the sequence of read commands etc.
seq:entity work.sequencer_ng 
	port map(
		reset_n							=> reset_n,
		clk								=> writeclk,
		runend							=> runend,
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
	terminate_output				<= '0';
	terminated_output				<= '0';
	header_counter					<= (others => '0');	
	subheader_counter				<= (others => '0');	
	debug_subheadercounter			<= (others => '0');
	debug_hitcounter				<= (others => '0');
elsif(writeclk'event and writeclk = '1') then
    noutoftime2 <= noutoftime;
    noverflow2  <= noverflow;
    nintime2    <= nintime;
    nout2       <= nout;

    out_ena							<= '0';
	out_is_hit						<= '0';
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
	
	if(conv_integer(readcommand_last3(COMMANDBITS-2 downto TIMESTAMPSIZE+4)) < NCHIPS) then
		memmultiplex						<= frommem(conv_integer(readcommand_last3(COMMANDBITS-2 downto TIMESTAMPSIZE+4)));
	end if;
	
	if(running_seq = '1') then
		tscounter <= tscounter + '1';
	end if;

	case readcommand_last4(COMMANDBITS-1 downto COMMANDBITS-4) is
	when COMMAND_HEADER1(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= tscounter(47 downto 16);
		out_type		<= MERGER_FIFO_PAKET_START_MARKER;
	when COMMAND_HEADER2(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= tscounter(15 downto 0) & header_counter;
		out_type		<= "0000";
		if(readcommand_ena_last4 = '1') then
 			header_counter <= header_counter + '1';
		end if;
	when COMMAND_DEBUGHEADER1(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= "0" & debug_subheadercounter(14 downto 0) & debug_hitcounter;
		out_type		<= "0000";
		debug_subheadercounter <= (others => '0');
		debug_hitcounter	   <= (others => '0');
	when COMMAND_DEBUGHEADER2(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= x"0AFEBABE";
		out_type		<= "0000";

	when COMMAND_SUBHEADER(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out		<= "111111" & "000" & readcommand_last4(TIMESTAMPSIZE-1 downto 4) & overflow_last4;
		out_type		<= "0000";
		if(readcommand_ena_last4 = '1') then
 			subheader_counter <= subheader_counter + '1';
			debug_subheadercounter <= debug_subheadercounter + '1';
		end if;
	when COMMAND_FOOTER(COMMANDBITS-1 downto COMMANDBITS-4) =>
		data_out 		<= header_counter & overflow_last4;
		out_type		<= MERGER_FIFO_PAKET_END_MARKER;
		if(runshutdown = '1')then
			terminate_output <= '1';
		end if;	
	when others =>
						-- ts(3:0) & chipID(5:0, the upper 2 bits are 0 here since we have max. 12 links at the moment)& row(7:0) & col(7:0) & tot(4:0) & '0'
		data_out		<= readcommand_last4(3 downto 0) & "00" & readcommand_last4(COMMANDBITS-2 downto TIMESTAMPSIZE+4) & memmultiplex & "0";
		out_type		<= "0000";
		out_is_hit		<= '1';
		if(readcommand_ena_last4 = '1') then
			nout <= nout + '1';
			debug_hitcounter <= debug_hitcounter + '1';
		end if;
	end case;
	if(terminate_output = '1') then
		data_out 		<= (others => '0');
		out_type		<= MERGER_FIFO_RUN_END_MARKER;
		out_ena			<= '1';
		terminate_output    <= '0';
		terminated_output	<= '1';
	end if;
	if(terminated_output = '1') then
		out_ena			<= '0';
	end if;	
end if;
end process;

credits32 <= conv_std_logic_vector(credits, 32);

e_mp_sorter_reg_mapping: entity work.mp_sorter_reg_mapping
    port map (
        i_clk156       => i_clk156,
        i_reset_n      => i_reset_n_regs,

        i_reg_add      => i_reg_add,
        i_reg_re       => i_reg_re,
        o_reg_rdata    => o_reg_rdata,
        i_reg_we       => i_reg_we,
        i_reg_wdata    => i_reg_wdata,

        i_nintime           => nintime2,
        i_noutoftime        => noutoftime2,
        i_noverflow         => noverflow2,
        i_nout              => nout2,
        i_credit            => credits32,
        o_sorter_delay      => delay--,
    );
end architecture RTL;
