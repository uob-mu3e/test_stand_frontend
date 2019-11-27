-- Stic/Mutrig event storing
-- Simon Corrodi, July 2017
-- corrodis@phys.ethz.ch
-- Konrad Briggl, April 2019: Stripped off pcie writer part, moved to separate file

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

--use work.mutrig_components.all;

entity mutrig_store is
port (
	i_clk_deser         : in  std_logic;
	i_clk_rd         : in  std_logic;					-- fast PCIe memory clk 
	i_reset          : in  std_logic;					-- reset, active low
	i_event_data     : in  std_logic_vector(47 downto 0);	-- event data from deserelizer
	i_event_ready    : in  std_logic;
	i_new_frame      : in  std_logic;					-- start of frame
	i_frame_info_rdy : in  std_logic;					-- frame info ready (2 cycles after new_frame)
	i_end_of_frame   : in  std_logic;					-- end   of frame
	i_frame_info     : in  std_logic_vector(15 downto 0);
	i_frame_number   : in  std_logic_vector(15 downto 0);
	i_crc_error      : in  std_logic;
--event data output inteface
	o_fifo_data	 : out std_logic_vector(55 downto 0);			-- event data output
	o_fifo_empty    :  out std_logic;
	i_fifo_rd	:  in std_logic;
--monitoring
	o_fifo_full       : out std_logic;					-- sync to i_clk_deser. Write-when-fill is prevented internally
	i_reset_counters : in std_logic;
	o_eventcounter   : out std_logic_vector(31 downto 0);			-- sync to i_clk_deser
	o_timecounter    : out std_logic_vector(63 downto 0);			-- sync to i_clk_deser
	o_framecounter    : out std_logic_vector(63 downto 0);			-- sync to i_clk_deser
	o_crcerrorcounter: out std_logic_vector(31 downto 0);			-- sync to i_clk_deser

	o_prbs_err_cnt   : out std_logic_vector(31 downto 0);
	o_prbs_wrd_cnt   : out std_logic_vector(63 downto 0);

--configuration
	i_SC_mask	: in std_logic						-- '1':  block any data from being written to the fifo
);
end mutrig_store;

architecture RTL of mutrig_store is
component channeldata_fifo is
	PORT
	(
		aclr		: IN STD_LOGIC  := '0';
		data		: IN STD_LOGIC_VECTOR (55 DOWNTO 0);
		rdclk		: IN STD_LOGIC ;
		rdreq		: IN STD_LOGIC ;
		wrclk		: IN STD_LOGIC ;
		wrreq		: IN STD_LOGIC ;
		q		: OUT STD_LOGIC_VECTOR (55 DOWNTO 0);
		rdempty		: OUT STD_LOGIC ;
		wrfull		: OUT STD_LOGIC;
		rdusedw		: OUT STD_LOGIC_VECTOR (7 DOWNTO 0)
	);
end component;

component prbs48_checker is
	port(
		i_clk		: in std_logic;
		i_new_cycle	: in std_logic; -- do not check next word, start checking after.
		i_rst_counter	: in std_logic; -- reset the error counter

		i_new_word	: in std_logic;
		i_prbs_word	: in std_logic_vector(47 downto 0);

		o_err_cnt	: out std_logic_vector(31 downto 0);
		o_wrd_cnt	: out std_logic_vector(63 downto 0)
	);
end component;


signal s_full_event_data	: std_logic_vector(55 downto 0);
signal s_event_ready		: std_logic;



-- fifo
signal s_fifofull                 : std_logic;
signal s_fifoused, s_fifoused_reg : std_logic_vector(7 downto 0);

signal s_fifofull_almost : std_logic;
--save data loss implementation
signal s_have_dropped    : std_logic;


-- counters
signal s_timecounter : std_logic_vector(63 downto 0);
signal s_eventcounter : std_logic_vector(31 downto 0);
signal s_framecounter : std_logic_vector(63 downto 0);
signal s_crcerrorcounter : std_logic_vector(31 downto 0);
begin 

u_prbs_checker : prbs48_checker
port map(
	i_clk		=> i_clk_deser,
	i_new_cycle	=> i_new_frame,
	i_rst_counter   => i_reset_counters,

	i_new_word	=> i_event_ready,
	i_prbs_word	=> i_event_data,

	o_err_cnt	=> o_prbs_err_cnt,
	o_wrd_cnt	=> o_prbs_wrd_cnt
);

pro_mux_event_data : process(i_clk_deser)
begin
if rising_edge(i_clk_deser) then
	s_fifoused_reg <= s_fifoused;
	if(s_fifoused_reg(7 downto 3)="1111") then
	       s_fifofull_almost <= '1';
	else
	       s_fifofull_almost <= '0';
	end if;

	--counters (event/time, errors/frame, errors/hit (prbs))
	if(i_reset_counters='1' or i_reset='1') then
		s_timecounter  <= (others=>'0');
		s_eventcounter <= (others=>'0');
		s_framecounter  <= (others=>'0');
		s_crcerrorcounter <= (others=>'0');
	else
		s_timecounter  <= std_logic_vector(unsigned(s_timecounter) + 1);
		if ( i_event_ready = '1' ) then
			s_eventcounter <= std_logic_vector(unsigned(s_eventcounter) + 1);
		end if;
		if ( i_end_of_frame = '1') then
			s_framecounter  <= std_logic_vector(unsigned(s_framecounter) + 1);
		end if;
		if ( i_end_of_frame = '1' and i_crc_error='1') then
			if(s_crcerrorcounter /= X"ffffffff") then
				s_crcerrorcounter <= std_logic_vector(unsigned(s_crcerrorcounter) + 1);
			end if;
		end if;
	end if;

	if i_reset = '1' then
		s_have_dropped  <='0';
		s_event_ready		<= '0';
	else

		if ( (i_event_ready = '1' or i_end_of_frame = '1' or i_frame_info_rdy= '1') and s_fifofull='0' and i_SC_mask='0') then
			s_event_ready		<= '1';
		else
			s_event_ready		<= '0';
		end if;

		--want to write hit data and fifo almost full, drop it (keeping flag)
		if( s_fifofull_almost='1' and i_end_of_frame='0' and i_frame_info_rdy='0' ) then
			s_event_ready<='0';
			s_have_dropped<='1';
		end if;

		--release have-dropped flag when seeing trailer
		if(i_end_of_frame='1') then
			s_have_dropped<='0';
		end if;

		--channel masked, drop any data
		if(i_SC_mask='1') then
			s_event_ready		<= '0';
		end if;
	end if;

	--selection of output data
	if(i_end_of_frame = '1' ) then 				-- the MSB of the event data is '0' for frame info data
		----------- TRAILER -----------
		s_full_event_data	<= "0000" & "11" & X"0000000"&"000"& s_have_dropped & i_frame_info(11) & i_crc_error & i_frame_number; -- identifier, hit-dropped-flag, l2 overflow, crc_error, frame id
	elsif i_frame_info_rdy= '1' then -- by defenition the first thing that happens
		----------- HEADER -----------
		s_full_event_data	<= "0000" & "10" & X"00000000" & "00" & i_frame_number; 
	elsif ( i_event_ready = '1' ) then		-- the MSB of the event data is '1' for event data)
		-----------  DATA  -----------
		--note: eflag reshuffled to have consistent position of this bit independent of data type
		-- identifier, short event flag, event data (cn,tbh,tcc,tf,ef,ebh,ecc,ef)
		if(i_frame_info(14)='1') then --short event
			s_full_event_data	<= "0000" & "00" &"0"& i_frame_info(14)  & i_event_data(47 downto 21) & 		  i_event_data(21 downto 1);
		else
			s_full_event_data	<= "0000" & "00" &"0"& i_frame_info(14)  & i_event_data(47 downto 22) & i_event_data(0) & i_event_data(21 downto 1);
		end if;
	end if;
end if;
end process;

u_channel_data_fifo : channeldata_fifo   
PORT MAP (
	aclr     => i_reset,
	data	   => s_full_event_data,
	rdclk	   => i_clk_rd,
	rdreq	   => i_fifo_rd,
	wrclk	   => i_clk_deser,
	wrreq	   => s_event_ready,
	q	      => o_fifo_data,
	rdempty	=> o_fifo_empty,
	wrfull	=> s_fifofull,
	rdusedw   => s_fifoused
);

o_fifo_full     <= s_fifofull;
o_eventcounter <= s_eventcounter;
o_crcerrorcounter <= s_crcerrorcounter;
o_timecounter <= s_timecounter;
o_framecounter <= s_framecounter;

end RTL;
