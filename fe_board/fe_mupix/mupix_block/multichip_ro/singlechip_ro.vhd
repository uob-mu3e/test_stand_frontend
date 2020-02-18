-----------------------------------
--
-- Single chip ro
-- take hits and coarse counters,
-- build events
-- Niklaus Berger, June 2015
-- 
-- niberger@uni-mainz.de
--
----------------------------------
 -- TODO: so far we take only lower 8b of TS, and no charge information

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_constants.all;
use work.daq_constants.all;

entity singlechip_ro is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= UNPACKER_HITSIZE
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		counter125			: in reg64;
		link_flag			: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR(HITSIZE-1 DOWNTO 0);
		hit_ena				: in STD_LOGIC;
		coarsecounter		: in STD_LOGIC_VECTOR(COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounter_ena	: in STD_LOGIC;
		chip_marker			: in byte_t;
		tomemdata			: out reg32;
		tomemena				: out std_logic;
		tomemeoe				: out std_logic
		);
end singlechip_ro;

architecture RTL of singlechip_ro is

type state_type is (IDLE, BEGINEVENT, EVCOUNT, TIMESTAMP1, TIMESTAMP2, OVERFLOW, CHIPCOUNTER, HITS_MSB, HITS_LSB, ENDEVENT);
signal NS : state_type;

signal coarsecounter_reg	: std_logic_vector(COARSECOUNTERSIZE-1 DOWNTO 0);
signal coarsecounter_ena_reg: std_logic;
signal hit_reg					: std_logic_vector(HITSIZE-1 DOWNTO 0);
signal hit_ena_reg			: std_logic;
signal eventcounter			: reg32;
signal hit_lsb					: reg32;

signal timestamp_reg			: reg64;

signal chip_marker_r		: byte_t;

begin

process(clk)
begin
if(rising_edge(clk))then
		chip_marker_r	<= chip_marker;
end if;
end process;

process(reset_n, clk)
begin
if(reset_n = '0') then
	tomemdata		<= (others => '0');
	tomemena 		<= '0';
	tomemeoe			<= '0';
	NS					<= IDLE;
	eventcounter	<= (others => '0');
	coarsecounter_reg	<= (others => '0');
	coarsecounter_ena_reg	<= '0';
	hit_reg				<= (others => '0');
	hit_ena_reg			<= '0';
	timestamp_reg		<= (others => '0');
	hit_lsb				<= (others => '0');
	
elsif(clk'event and clk = '1') then

	tomemena 		<= '0';
	tomemeoe			<= '0';

	if(link_flag = '1')then
		timestamp_reg			<= counter125;
	end if;
	
	if(coarsecounter_ena = '1') then
		coarsecounter_reg		<= coarsecounter;
		coarsecounter_ena_reg<= '1';
	end if;
	
	if(hit_ena = '1' and coarsecounter_ena = '0')then -- and hit_ena_reg = '0') then
		hit_reg		<= hit_in;
		hit_ena_reg <= '1';
	end if;

    case NS is
        when IDLE =>
            if(link_flag = '1') then
                --NS <= BEGINEVENT;
                NS              <= TIMESTAMP1;
                hit_ena_reg	    <= '0';-- no hit stored!
            end if;
--      when BEGINEVENT =>      -- 1 or 2 after link_flag
--      tomemdata           <= BEGINOFEVENT;
--      tomemena            <= '1';
--      NS                  <= EVCOUNT;
        when TIMESTAMP1 =>      -- 3 after link_flag or 4: counter might be available
            tomemdata           <= timestamp_reg(47 downto 16);	-- for now: send FPGA timestamp
            tomemena            <= '1';
            NS                  <= TIMESTAMP2;
        when TIMESTAMP2 =>      -- 4 after link_flag: counter might be available (timerend = 0 and hits available) or 1 after cnt_ena
            tomemdata           <= timestamp_reg(15 downto 0) & x"0000";	-- for now: send FPGA timestamp
            tomemena            <= '1';
            NS                  <= EVCOUNT;
            
        when EVCOUNT =>				-- 2 or 3 after link_flag
			tomemdata			<= '0' & eventcounter(30 downto 0);		-- counts all blocks that are produced by this entity
			eventcounter		<= eventcounter + '1';
			tomemena				<= '1';
			NS						<= OVERFLOW;
		when OVERFLOW	=>				-- 1 or 2 after cnt_ena 
			tomemdata			<= (others => '0');	-- here some useful information can be transferred
			tomemena				<= '1';
			NS						<= CHIPCOUNTER;		
		when CHIPCOUNTER =>			-- 2 or 3 after cnt_ena - here we have to check if a counter was seen, then we proceed accordingly!
											-- if counter is seen, go on, otherwise go to End of block because no hits are incoming!
			if(coarsecounter_ena_reg = '1')then
				tomemdata 			<= '0' & chip_marker_r(4 downto 0) & "00" & coarsecounter_reg (23 downto 0);
				tomemena				<= '1';
				NS						<= HITS_MSB;
				coarsecounter_ena_reg	<= '0';
			elsif(link_flag = '1')then
				tomemdata 			<= '0' & chip_marker_r(4 downto 0) & "11" & x"001337";
				tomemena				<= '1';
				NS						<= ENDEVENT;
				coarsecounter_ena_reg	<= '0';			
			end if;
		when HITS_MSB =>				-- 3 after cnt_ena: before the next hit, and also before the next counter or hit_ena might be on
			if(link_flag = '1') then
				NS					<= ENDEVENT;
			elsif(hit_ena_reg = '1') then
				tomemdata		<= '0' & chip_marker_r(4 downto 0) & "00" & HITLABEL & hit_reg(35 downto 32) & hit_reg(31 downto 16);
				-- chip(5b), hitlabel(4b), LinkID(4b), Row(8b), Col(8b)
				tomemena			<= '1';
				if(hit_ena = '0')then 
					hit_ena_reg		<= '0';	-- clear this bit!		
				end if;
				hit_lsb	<= '0' & chip_marker_r(4 downto 0) & "00" & TIMELABEL & hit_reg(35 downto 32) & hit_reg(15 downto 0);
				-- to assure that we don't mix up hit information in the next state!
				NS					<= HITS_LSB;
			end if;
		when HITS_LSB =>				-- 4 after hit_ena: registers the next hit!
			NS					<= HITS_MSB;
			tomemdata		<= hit_lsb;--'0' & chip_marker_r(4 downto 0) & "00" & TIMELABEL & hit_reg(35 downto 32) & hit_reg(15 downto 0);
			-- chip(5b), timelabel(4b), LinkID(4b), Charge(6b) & TS(10b)
			tomemena			<= '1';
			
		when ENDEVENT =>
			tomemdata			<= ENDOFEVENT;
			tomemena				<= '1';
			tomemeoe				<= '1';
			NS						<= BEGINEVENT;
		when others =>
			NS 					<= IDLE;
	end case;	
end if;
end process;


end rtl;