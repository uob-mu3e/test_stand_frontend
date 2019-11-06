-----------------------------------
--
-- Multi chip ro
-- relies on single chip RO,
-- used in parallel
-- Sebastian Dittmeier, July 2015
-- 
-- dittmeier@physi.uni-heidelberg.de
--
-- adapted to MP8
-- April 2017
----------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.datapath_components.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity multichip_ro_zerosupressed is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= 40;
		NCHIPS 				: integer 	:= 16		
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		counter125			: in reg64;
		link_flag			: in std_logic_vector(NCHIPS-1 downto 0);
		hits_in				: in STD_LOGIC_VECTOR (NCHIPS*HITSIZE-1 DOWNTO 0);
		hits_ena				: in STD_LOGIC_VECTOR (NCHIPS-1 downto 0);
		coarsecounters		: in STD_LOGIC_VECTOR (NCHIPS*COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounters_ena: in STD_LOGIC_VECTOR (NCHIPS-1 downto 0);
--		error_flags			: in STD_LOGIC_VECTOR (NCHIPS*4-1 downto 0);		
		chip_markers		: in reg32;
		prescale				: in STD_LOGIC_VECTOR(31 downto 0);
		is_shared			: in std_logic_vector(NCHIPS-1 downto 0);
		tomemdata			: out reg32;
		tomemena				: out std_logic;
		tomemeoe				: out std_logic;
		errcounter_overflow : out reg32;
		errcounter_sel_in		: in std_logic_vector(CHIPRANGE-1 downto 0);
		readtrigfifo		: out std_logic;
		fromtrigfifo		: in reg64;
		trigfifoempty		:	 in std_logic;
		readhbfifo: out std_logic;
		fromhbfifo: in reg64;
		hbfifoempty: in std_logic				
		);
end multichip_ro_zerosupressed;
		
architecture RTL of multichip_ro_zerosupressed is


signal 	singlezs2fifo		: links_reg32;
signal 	singlezs2fifoena	: std_logic_vector(NCHIPS-1 downto 0);
--signal 	singlezs2fifoena_last	: std_logic_vector(NCHIPS-1 downto 0);
signal 	singlezs2fifoeoe	: std_logic_vector(NCHIPS-1 downto 0);
signal 	fifo_rdreq			: std_logic_vector(NCHIPS-1 downto 0);
--signal 	fifo_empty			: std_logic_vector(NCHIPS-1 downto 0);
--signal 	fifo_full			: std_logic_vector(NCHIPS-1 downto 0);
signal 	fifo_q				: links_vec33;
signal 	fifo_usedw			: links_vec9;
signal	drop_block			: std_logic_vector(NCHIPS-1 downto 0);

signal 	fifo_wrreq			: std_logic_vector(NCHIPS-1 downto 0);

signal 	fifo_data			: links_vec33;
signal 	fifo_q_reg			: links_vec33;

signal 	fifo_eoe_cnt		: links_vec8;
signal 	roundrobin			: integer range 0 to NCHIPS+1;	-- include triggers and hitbus as new chip
signal 	fifo_aclr			: std_logic;

signal 	fifo_overflow		: links_reg32;

signal 	errcounter_sel	: std_logic_vector(2 downto 0);

type ro_type is (INCREASE, WAITING, WAITING_FAST, INIT_READ, READING, FINISH_READ, WAITING_TRIGGER, TRIGGERHEADER, TRIGGERDATA_MSB, TRIGGERDATA_LSB, TRIGGERFOOTER, WAITING_HB, HBHEADER, HBDATA_MSB, HBDATA_LSB, HBFOOTER);
signal 	multi_ro : ro_type;

signal 	triggercounter : reg32;
signal 	hbcounter		: reg32;

signal	prescale_r		: reg32;
signal 	marker 			: links_vec8;

begin

fifo_aclr <= not reset_n;

process(clk)
begin
if(rising_edge(clk))then
	prescale_r				<= prescale;
end if;
end process;	

time_demux_zs:
for i in 0 to NCHIPS-1 generate
	
	marker(i)	<= std_logic_vector(to_unsigned(i,8));

-- experimental: move to timemuxed RO for links 3, 7, 11 and 15
	MUXED_LINKS: if ((i mod 4) = 3) generate 
	
	timemuxed_ro: timemuxed_ro_zerosupressed 
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE				=> HITSIZE,
			NCHIPS 				=> 3	
		)
		port map(
			reset_n				=> reset_n,
			clk					=> clk,
			counter125			=> counter125,
			link_flag			=> link_flag(i),
			hit_in				=> hits_in(HITSIZE*(i+1)-1 downto HITSIZE*i),
			hit_ena				=> hits_ena(i),
			coarsecounter		=> coarsecounters(COARSECOUNTERSIZE*(i+1)-1 downto COARSECOUNTERSIZE*i),
			coarsecounter_ena	=> coarsecounters_ena(i),
			is_shared			=> is_shared(i),
--			error_flags			=> error_flags(i*4+3 downto i*4),			
			chip_marker			=> marker(i),--chip_markers(CHIPMARKERSIZE*(i+1)-1 downto CHIPMARKERSIZE*i),	-- what to do here?!
			prescale				=> prescale_r,	
			tomemdata			=> singlezs2fifo(i),
			tomemena				=> singlezs2fifoena(i),
			tomemeoe				=> singlezs2fifoeoe(i)
			);
	end generate MUXED_LINKS;	
	
-- these here are definitely single links	
	SINGLE_LINKS: if ((i mod 4) /= 3) generate 
	
	singlechip_ro: singlechip_ro_zerosupressed
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE				=> HITSIZE
		)
		port map(
			reset_n				=> reset_n,
			clk					=> clk,
			counter125			=> counter125,
			link_flag			=> link_flag(i),
			hit_in				=> hits_in(HITSIZE*(i+1)-1 downto HITSIZE*i),
			hit_ena				=> hits_ena(i),
			coarsecounter		=> coarsecounters(COARSECOUNTERSIZE*(i+1)-1 downto COARSECOUNTERSIZE*i),
			coarsecounter_ena	=> coarsecounters_ena(i),	
			chip_marker			=> marker(i),--chip_markers(CHIPMARKERSIZE*(i+1)-1 downto CHIPMARKERSIZE*i),	-- what to do here?!
			prescale				=> prescale_r,	
			tomemdata			=> singlezs2fifo(i),
			tomemena				=> singlezs2fifoena(i),
			tomemeoe				=> singlezs2fifoeoe(i)
		);	
	end generate SINGLE_LINKS;	
	
			
	fifo_writing: process(clk, reset_n)
	begin
		if(reset_n = '0')then
			fifo_data(i) 		<= (others => '0');
			fifo_wrreq(i)		<= '0';
			fifo_overflow(i) 	<= (others => '0');
--			singlezs2fifoena_last(i) <= '0';	
			drop_block(i)		<= '0';
		elsif(rising_edge(clk))then
			fifo_data(i) 		<= singlezs2fifoeoe(i) & singlezs2fifo(i);
			if(drop_block(i) = '1') then
				fifo_wrreq(i) 		<= '0';
				if(singlezs2fifoena(i)='1')then
					fifo_overflow(i) 	<= fifo_overflow(i) + 1 ;
					if(singlezs2fifoeoe(i) = '1') then -- block must be finished! -- or singlezs2fifoena(i)='0')then
						drop_block(i)	<= '0';
					end if;					
				end if;	
			elsif(fifo_usedw(i) > "110011001" and singlezs2fifoena(i)='1' and singlezs2fifo(i)=BEGINOFEVENT)then	--if more than  409 words are written, drop!
				fifo_wrreq(i) 		<= '0';
				fifo_overflow(i) 	<= fifo_overflow(i) + 1 ;	
				drop_block(i)		<= '1';
			else
				fifo_wrreq(i) 		<= singlezs2fifoena(i);						
			end if;
			
		end if;
	end process fifo_writing;
	
		
	fifo_evtcnt : process	(clk, reset_n)
	begin
		if(reset_n = '0')then
			fifo_eoe_cnt(i)	<= (others => '0');
		elsif(rising_edge(clk))then
			if(fifo_data(i)(32)='1' and fifo_eoe_cnt(i) < x"FF" and fifo_wrreq(i)='1')then	-- error handling!
				fifo_eoe_cnt(i) <= fifo_eoe_cnt(i) + 1;
				if(fifo_q(i)(32)='1' and fifo_rdreq(i) = '1')then
					fifo_eoe_cnt(i) <= fifo_eoe_cnt(i);
				end if;
			else
				if(fifo_q(i)(32)='1' and fifo_eoe_cnt(i)>x"00" and fifo_rdreq(i) = '1')then
					fifo_eoe_cnt(i) 	<= fifo_eoe_cnt(i) - 1;
				end if;				
			end if;
		end if;
	end process fifo_evtcnt;	
	
	
	all_singlerozs_fifos : single_zs_fifo 
		PORT MAP
		(
			aclr		=> fifo_aclr,
			clock		=> clk,
			data		=> fifo_data(i),
			rdreq		=> fifo_rdreq(i),
			wrreq		=> fifo_wrreq(i),
			empty		=> open,--fifo_empty(i),
			full		=> open,--fifo_full(i),
			q			=> fifo_q(i),
			usedw		=> fifo_usedw(i)
		);	
		
	read_clocking: process(clk, reset_n)
	begin
	if(reset_n = '0')then
		fifo_q_reg(i) 		<= (others => '0');
	elsif(rising_edge(clk))then
		fifo_q_reg(i)	 	<= fifo_q(i);
	end if;
	end process read_clocking;
		
end generate;	


	fifo_roundrobin : process	(clk, reset_n)
	
	variable success : std_logic := '0';	
	
	begin
		if(reset_n = '0')then
			roundrobin		<= 0;
			fifo_rdreq		<= (others => '0');
			tomemena		 	<= '0';
			tomemdata		<= (others => '0');	
			tomemeoe			<= '0';	
			triggercounter	<= (others => '0');
			readtrigfifo	<= '0';	
			hbcounter		<= (others => '0');			
			readhbfifo		<= '0';		
			multi_ro			<= WAITING_FAST;--INCREASE;
			success			:= '0';
		elsif(rising_edge(clk))then					

			tomemena		<= '0';
			tomemdata	<= (others => '0');	
			tomemeoe		<= '0';			

			case multi_ro is
			
				when INCREASE	=>
					tomemena		<= '0';
					tomemdata	<= (others => '0');	
					tomemeoe		<= '0';		
					if(roundrobin < NCHIPS+1)then
						roundrobin	<= roundrobin + 1;
					else
						roundrobin	<= 0;						
					end if;
					if(roundrobin 		< NCHIPS-1)then
						multi_ro 	<= WAITING;
					elsif(roundrobin	= NCHIPS-1)then
						multi_ro 	<= WAITING_TRIGGER;
					elsif(roundrobin 	= NCHIPS)then
						multi_ro 	<= WAITING_HB;	
					elsif(roundrobin 	= NCHIPS+1)then
						multi_ro 	<= WAITING;							
					end if;
					
				when WAITING_FAST =>
					for I in 0 to NCHIPS+1 loop
						if(I >= roundrobin) then
							if(I < NCHIPS)then
								if (fifo_eoe_cnt(I) /= 0 and success = '0') then
									fifo_rdreq(I)	<= '1';				
									multi_ro			<= INIT_READ;	
									roundrobin		<= I;
									success			:= '1';
								end if;
							elsif(I = NCHIPS)then
								if(trigfifoempty = '0' and success = '0') then					
									tomemdata	 <= BEGINOFTRIGGER;					
									tomemena		 <= '1';	
									multi_ro		 <= TRIGGERHEADER;
									readtrigfifo <= '1';			
									roundrobin	 <= I;	
									success		 := '1';								
								end if;						
							elsif(I = NCHIPS+1)then
								if(hbfifoempty = '0' and success = '0') then					
									tomemdata	 <= BEGINOFHB;					
									tomemena		 <= '1';	
									multi_ro		 <= HBHEADER;
									readhbfifo 	 <= '1';	
									roundrobin	 <= I;
									success		 := '1';								
								end if;						
							end if;
						end if;
					end loop;
					if(success = '0')then
						roundrobin	<= 0;
					end if;
--					if(success = '0')then
--						for I in 0 to NCHIPS+1 loop
--							if (I <= roundrobin) then
--								if(I < NCHIPS)then
--									if (fifo_eoe_cnt(I) /= 0 and success = '0') then
--										fifo_rdreq(I)	<= '1';				
--										multi_ro			<= INIT_READ;	
--										roundrobin		<= I;
--										success			:= '1';
--									end if;
--								elsif(I = NCHIPS)then
--									if(trigfifoempty = '0' and success = '0') then					
--										tomemdata	 <= BEGINOFTRIGGER;					
--										tomemena		 <= '1';	
--										multi_ro		 <= TRIGGERHEADER;
--										readtrigfifo <= '1';			
--										roundrobin	 <= I;	
--										success		:= '1';								
--									end if;						
--								elsif(I = NCHIPS+1)then
--									if(hbfifoempty = '0' and success = '0') then					
--										tomemdata	 <= BEGINOFHB;					
--										tomemena		 <= '1';	
--										multi_ro		 <= HBHEADER;
--										readhbfifo 	 <= '1';	
--										roundrobin	 <= I;
--										success		:= '1';								
--									end if;						
--								end if;
--							end if;
--						end loop;					
--					end if;
					
-------------- MUPIX DATA					
					
				when WAITING	=>
					if(fifo_eoe_cnt(roundrobin) = 0)then
						multi_ro 	<= INCREASE;
					else
						fifo_rdreq(roundrobin)	<= '1';				
						multi_ro		<= INIT_READ;
					end if;
					
				when INIT_READ	=>
					success			:= '0';
					multi_ro 		<= READING;
					
				when READING	=>
					tomemdata		<= fifo_q_reg(roundrobin)(31 downto 0);
					tomemeoe			<= fifo_q_reg(roundrobin)(32);
					tomemena			<= '1';
					if(fifo_q(roundrobin)(32)='1')then
						multi_ro		<= FINISH_READ;
						fifo_rdreq(roundrobin)	<= '0';						
					end if;
					
				when FINISH_READ	=>
					tomemdata		<= fifo_q_reg(roundrobin)(31 downto 0);
					tomemeoe			<= fifo_q_reg(roundrobin)(32);
					tomemena			<= '1';
					multi_ro			<= WAITING_FAST;--INCREASE;
					roundrobin		<= roundrobin + 1;				-- only for fast mode


---------------TRIGGER	
					
				when WAITING_TRIGGER  =>
					if(trigfifoempty = '1') then					
						multi_ro 	<= INCREASE;
					else
						tomemdata	 <= BEGINOFTRIGGER;					
						tomemena		 <= '1';	
						multi_ro		 <= TRIGGERHEADER;
						readtrigfifo <= '1';						
					end if;
				
				when TRIGGERHEADER =>
					success				:= '0';				
					readtrigfifo 		<= '0';				
					tomemdata			<= '0' & triggercounter(30 downto 0);
					tomemena				<= '1';
					triggercounter 	<= triggercounter + '1';
					multi_ro				<= TRIGGERDATA_MSB;

				when TRIGGERDATA_MSB =>
					tomemdata		<= fromtrigfifo(REG64_TOP_RANGE);
					tomemena			<= '1';
					if(trigfifoempty = '1')then
						readtrigfifo 	<= '0';
					else
						readtrigfifo 	<= '1';					
					end if;
					multi_ro 		<= TRIGGERDATA_LSB;
					
				when TRIGGERDATA_LSB =>
					tomemdata		<= fromtrigfifo(REG64_BOTTOM_RANGE);
					tomemena			<= '1';
					readtrigfifo 	<= '0';						
					if(trigfifoempty = '1')then
						multi_ro 		<= TRIGGERFOOTER;
					else
						multi_ro 		<= TRIGGERDATA_MSB;						
					end if;					
					
				when TRIGGERFOOTER =>
					-- End of triggers, go back to start
					tomemdata 		<= ENDOFTRIGGER;
					tomemena			<= '1';
					tomemeoe			<= '1';
					multi_ro			<= WAITING_FAST;--INCREASE;	
					roundrobin		<= roundrobin + 1;				-- only for fast mode					

					
---------------HITBUS

				when WAITING_HB =>
					if(hbfifoempty = '1') then					
						multi_ro 	<= INCREASE;
					else
						tomemdata	 <= BEGINOFHB;					
						tomemena		 <= '1';	
						multi_ro		 <= HBHEADER;
						readhbfifo 	 <= '1';						
					end if;
						
				when HBHEADER =>
					success				:= '0';					
					readhbfifo 			<= '0';				
					tomemdata			<= '0' & hbcounter(30 downto 0);
					tomemena				<= '1';
					hbcounter 			<= hbcounter + '1';
					multi_ro				<= HBDATA_MSB;

				when HBDATA_MSB =>
					tomemdata		<= fromhbfifo(REG64_TOP_RANGE);
					tomemena			<= '1';
					if(hbfifoempty = '1')then
						readhbfifo 	<= '0';
					else
						readhbfifo 	<= '1';					
					end if;
					multi_ro 		<= HBDATA_LSB;
					
				when HBDATA_LSB =>
					tomemdata		<= fromhbfifo(REG64_BOTTOM_RANGE);
					tomemena			<= '1';
					readhbfifo 		<= '0';						
					if(hbfifoempty = '1')then
						multi_ro 	<= HBFOOTER;
					else
						multi_ro 	<= HBDATA_MSB;						
					end if;					
					
				when HBFOOTER =>
					-- End of triggers, go back to start
					tomemdata 		<= ENDOFHB;
					tomemena			<= '1';
					tomemeoe			<= '1';
					multi_ro			<= WAITING_FAST;--INCREASE;
					roundrobin		<= 0;				-- only for fast mode					
						
				when others	=>
					multi_ro 	<= WAITING_FAST;--INCREASE;
					tomemena		<= '0';
					tomemdata	<= (others => '0');	
					tomemeoe		<= '0';	
					roundrobin	<= 0;				-- only for fast mode						
					
			end case;
		end if;
	end process fifo_roundrobin;
	


process(clk)
begin	-- clocked mux
	if(rising_edge(clk))then
		errcounter_overflow	<= fifo_overflow(to_integer(unsigned(errcounter_sel_in)));
	end if;
end process;



end RTL;
