-----------------------------------
-- Time muxed ro with zero suppression
-- Sebastian Dittmeier, June 2017
-- dittmeier@physi.uni-heidelberg.de
--
-- Idea behind this entity:
-- If we use a time multiplexed MuPix link
-- We get 4 clock cycles of data per link
-- before we change to the next link
-- so we simply count to 4
-- and cycle through 3 RO entities
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_types.all;
use work.mupix_constants.all;

entity timemuxed_ro_zerosupressed is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= 40;
		NCHIPS 				: integer 	:= 3		
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		counter125			: in reg64;
		link_flag			: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
		hit_ena				: in STD_LOGIC;
		coarsecounter		: in STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounter_ena	: in STD_LOGIC;
--		error_flags			: in STD_LOGIC_VECTOR(3 downto 0);		
		chip_marker			: in chipmarkertype;
		prescale				: in STD_LOGIC_VECTOR(31 downto 0);
		is_shared			: in std_logic;
		tomemdata			: out reg32;
		tomemena				: out std_logic;
		tomemeoe				: out std_logic			
		);
end timemuxed_ro_zerosupressed;

architecture RTL of timemuxed_ro_zerosupressed is

signal 	time_demux			: std_logic_vector(1 downto 0);
signal 	time_ena				: std_logic_vector(NCHIPS-1 downto 0);

type mux_reg32 is array(NCHIPS-1 downto 0) of reg32;
type mux_reg33 is array(NCHIPS-1 downto 0) of std_logic_vector(32 downto 0);
type mux_reg9 is array(NCHIPS-1 downto 0) of std_logic_vector(8 downto 0);
type mux_reg8 is array(NCHIPS-1 downto 0) of std_logic_vector(7 downto 0);

signal 	singlezs2fifo		: mux_reg32;
signal 	singlezs2fifoena	: std_logic_vector(NCHIPS-1 downto 0);
--signal 	singlezs2fifoena_last	: std_logic_vector(NCHIPS-1 downto 0);
signal 	singlezs2fifoeoe	: std_logic_vector(NCHIPS-1 downto 0);
signal 	fifo_rdreq			: std_logic_vector(NCHIPS-1 downto 0);
--signal 	fifo_empty			: std_logic_vector(NCHIPS-1 downto 0);
--signal 	fifo_full			: std_logic_vector(NCHIPS-1 downto 0);
signal 	fifo_q				: mux_reg33;
signal 	fifo_usedw			: mux_reg9;
signal	drop_block			: std_logic_vector(NCHIPS-1 downto 0);

signal 	fifo_wrreq			: std_logic_vector(NCHIPS-1 downto 0);

signal	hit_in_reg				:  STD_LOGIC_VECTOR(HITSIZE-1 DOWNTO 0);
signal	coarsecounter_reg		:  STD_LOGIC_VECTOR(COARSECOUNTERSIZE-1 DOWNTO 0);
signal	link_flag_reg			:  std_logic_vector(NCHIPS-1 downto 0);
signal	hit_ena_reg				:  std_logic_vector(NCHIPS-1 downto 0);
signal	coarsecounter_ena_reg:  std_logic_vector(NCHIPS-1 downto 0);

signal 	fifo_data			: mux_reg33;
signal 	fifo_q_reg			: mux_reg33;

signal 	fifo_eoe_cnt		: mux_reg8;
signal 	roundrobin			: integer range 0 to NCHIPS+1;	-- include triggers and hitbus as new chip
signal 	fifo_aclr			: std_logic;

signal 	fifo_overflow		: mux_reg33;

signal 	errcounter_sel	: std_logic_vector(2 downto 0);

type ro_type is (INCREASE, WAITING, WAITING_FAST, INIT_READ, READING, FINISH_READ);
signal 	multi_ro : ro_type;

signal	prescale_r		: reg32;

begin

fifo_aclr <= not reset_n;

process(clk)
begin
if(rising_edge(clk))then
	prescale_r				<= prescale;
end if;
end process;	

	
		
time_demuxing : process(clk, reset_n)
begin
	if(reset_n = '0')then
		time_demux	<= (others => '0');
		hit_in_reg	<= (others => '0');
		coarsecounter_reg	<= (others => '0');
		link_flag_reg		<= (others => '0');
		hit_ena_reg			<= (others => '0');
		coarsecounter_ena_reg	<= (others => '0');
		time_ena				<= (others => '0');
	elsif(rising_edge(clk))then
	
		hit_in_reg					<= hit_in;
		coarsecounter_reg			<= coarsecounter;
		for k in 0 to NCHIPS-1 loop
			link_flag_reg(k)				<= link_flag and time_ena(k);
			hit_ena_reg(k)					<= hit_ena and time_ena(k);
			coarsecounter_ena_reg(k)	<= coarsecounter_ena and time_ena(k);			
		end loop;
		time_demux <= time_demux + "01";
		if(time_demux = "00")then
			if(time_ena = 0)then	-- reset condition and catch failure
				time_ena 	<= (others => '0');		
				time_ena(0) <= '1';
			else
				if(is_shared = '1')then	-- otherwise we just stay with RO #1
					time_ena <= time_ena(NCHIPS-2 downto 0) & time_ena(NCHIPS-1);
				end if;
			end if;
		end if;

	end if;
end process time_demuxing;

all_zerosuppressed:
for i in 0 to NCHIPS-1 generate	

	all_singlerozs : work.singlechip_ro_zerosupressed 
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE				=> HITSIZE
		)
		port map(
			reset_n				=> reset_n,
			clk					=> clk,
			counter125			=> counter125,
			link_flag			=> link_flag_reg(i),
			hit_in				=> hit_in_reg,
			hit_ena				=> hit_ena_reg(i),
			coarsecounter		=> coarsecounter_reg,
			coarsecounter_ena	=> coarsecounter_ena_reg(i),
--			error_flags			=> error_flags(i*4+3 downto i*4),			
			chip_marker			=> chip_marker,--s(CHIPMARKERSIZE*(i+1)-1 downto CHIPMARKERSIZE*i),	-- what to do here?!
			prescale				=> prescale_r,	
			tomemdata			=> singlezs2fifo(i),
			tomemena				=> singlezs2fifoena(i),
			tomemeoe				=> singlezs2fifoeoe(i)
			);
			
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
	
	
	all_singlerozs_fifos : work.single_zs_fifo 
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
--			triggercounter	<= (others => '0');
--			readtrigfifo	<= '0';	
--			hbcounter		<= (others => '0');			
--			readhbfifo		<= '0';		
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
--					elsif(roundrobin	= NCHIPS-1)then
--						multi_ro 	<= WAITING_TRIGGER;
--					elsif(roundrobin 	= NCHIPS)then
--						multi_ro 	<= WAITING_HB;	
--					elsif(roundrobin 	= NCHIPS+1)then
--						multi_ro 	<= WAITING;							
					end if;
					
				when WAITING_FAST =>
					for I in 0 to NCHIPS-1 loop
						if(I >= roundrobin) then
							if(I < NCHIPS)then
								if (fifo_eoe_cnt(I) /= 0 and success = '0') then
									fifo_rdreq(I)	<= '1';				
									multi_ro			<= INIT_READ;	
									roundrobin		<= I;
									success			:= '1';
								end if;
--							elsif(I = NCHIPS)then
--								if(trigfifoempty = '0' and success = '0') then					
--									tomemdata	 <= BEGINOFTRIGGER;					
--									tomemena		 <= '1';	
--									multi_ro		 <= TRIGGERHEADER;
--									readtrigfifo <= '1';			
--									roundrobin	 <= I;	
--									success		 := '1';								
--								end if;						
--							elsif(I = NCHIPS+1)then
--								if(hbfifoempty = '0' and success = '0') then					
--									tomemdata	 <= BEGINOFHB;					
--									tomemena		 <= '1';	
--									multi_ro		 <= HBHEADER;
--									readhbfifo 	 <= '1';	
--									roundrobin	 <= I;
--									success		 := '1';								
--								end if;						
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
--					
--				when WAITING_TRIGGER  =>
--					if(trigfifoempty = '1') then					
--						multi_ro 	<= INCREASE;
--					else
--						tomemdata	 <= BEGINOFTRIGGER;					
--						tomemena		 <= '1';	
--						multi_ro		 <= TRIGGERHEADER;
--						readtrigfifo <= '1';						
--					end if;
--				
--				when TRIGGERHEADER =>
--					success				:= '0';				
--					readtrigfifo 		<= '0';				
--					tomemdata			<= '0' & triggercounter(30 downto 0);
--					tomemena				<= '1';
--					triggercounter 	<= triggercounter + '1';
--					multi_ro				<= TRIGGERDATA_MSB;
--
--				when TRIGGERDATA_MSB =>
--					tomemdata		<= fromtrigfifo(REG64_TOP_RANGE);
--					tomemena			<= '1';
--					if(trigfifoempty = '1')then
--						readtrigfifo 	<= '0';
--					else
--						readtrigfifo 	<= '1';					
--					end if;
--					multi_ro 		<= TRIGGERDATA_LSB;
--					
--				when TRIGGERDATA_LSB =>
--					tomemdata		<= fromtrigfifo(REG64_BOTTOM_RANGE);
--					tomemena			<= '1';
--					readtrigfifo 	<= '0';						
--					if(trigfifoempty = '1')then
--						multi_ro 		<= TRIGGERFOOTER;
--					else
--						multi_ro 		<= TRIGGERDATA_MSB;						
--					end if;					
--					
--				when TRIGGERFOOTER =>
--					-- End of triggers, go back to start
--					tomemdata 		<= ENDOFTRIGGER;
--					tomemena			<= '1';
--					tomemeoe			<= '1';
--					multi_ro			<= WAITING_FAST;--INCREASE;	
--					roundrobin		<= roundrobin + 1;				-- only for fast mode					
--
--					
-----------------HITBUS
--
--				when WAITING_HB =>
--					if(hbfifoempty = '1') then					
--						multi_ro 	<= INCREASE;
--					else
--						tomemdata	 <= BEGINOFHB;					
--						tomemena		 <= '1';	
--						multi_ro		 <= HBHEADER;
--						readhbfifo 	 <= '1';						
--					end if;
--						
--				when HBHEADER =>
--					success				:= '0';					
--					readhbfifo 			<= '0';				
--					tomemdata			<= '0' & hbcounter(30 downto 0);
--					tomemena				<= '1';
--					hbcounter 			<= hbcounter + '1';
--					multi_ro				<= HBDATA_MSB;
--
--				when HBDATA_MSB =>
--					tomemdata		<= fromhbfifo(REG64_TOP_RANGE);
--					tomemena			<= '1';
--					if(hbfifoempty = '1')then
--						readhbfifo 	<= '0';
--					else
--						readhbfifo 	<= '1';					
--					end if;
--					multi_ro 		<= HBDATA_LSB;
--					
--				when HBDATA_LSB =>
--					tomemdata		<= fromhbfifo(REG64_BOTTOM_RANGE);
--					tomemena			<= '1';
--					readhbfifo 		<= '0';						
--					if(hbfifoempty = '1')then
--						multi_ro 	<= HBFOOTER;
--					else
--						multi_ro 	<= HBDATA_MSB;						
--					end if;					
--					
--				when HBFOOTER =>
--					-- End of triggers, go back to start
--					tomemdata 		<= ENDOFHB;
--					tomemena			<= '1';
--					tomemeoe			<= '1';
--					multi_ro			<= WAITING_FAST;--INCREASE;
--					roundrobin		<= 0;				-- only for fast mode					
--						
--				when others	=>
--					multi_ro 	<= WAITING_FAST;--INCREASE;
--					tomemena		<= '0';
--					tomemdata	<= (others => '0');	
--					tomemeoe		<= '0';	
--					roundrobin	<= 0;				-- only for fast mode						
					
			end case;
		end if;
	end process fifo_roundrobin;
	


--process(clk)
--begin	-- clocked mux
--	if(rising_edge(clk))then
--		errcounter_overflow	<= fifo_overflow(to_integer(unsigned(errcounter_sel_in)));
--	end if;
--end process;



end RTL;
