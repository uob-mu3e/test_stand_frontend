-----------------------------------
-- Single chip ro with zero suppression
-- take hits and coarse counters,
-- build events
-- Niklaus Berger, June 2015
-- niberger@uni-mainz.de
--
-- Updated for MP8 and adapted SingleRO
-- Sebastian Dittmeier, April 2017
-- dittmeier@physi.uni-heidelberg.de
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.datapath_components.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity singlechip_ro_zerosupressed is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= UNPACKER_HITSIZE
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
		tomemdata			: out reg32;
		tomemena				: out std_logic;
		tomemeoe				: out std_logic
		);
end singlechip_ro_zerosupressed;

architecture RTL of singlechip_ro_zerosupressed is

type state_type is (IDLE, BEGINEVENT, EVCOUNT, CHIPCOUNTER, TIMESTAMP1, TIMESTAMP2, HITS, ENDEVENT,
							BEGINEVENT_ERROR, HITS_ERROR);
signal NS : state_type;

signal coarsecounter_reg	: std_logic_vector (COARSECOUNTERSIZE-1 DOWNTO 0);
signal eventcounter			: reg32;

signal timestamp_reg			: reg64;

constant NDELAY : integer := 7;		-- in case of error in counter but no hit
type delayarray_t is array (NDELAY-1 downto 0) of reg32;
signal delayarray : delayarray_t;
signal enadelay : std_logic_vector(NDELAY-1 downto 0);
signal eoedelay : std_logic_vector(NDELAY-1 downto 0);

signal count_ena : std_logic_vector(3 downto 0);

--signal hitspresent : std_logic;
signal todelaydata : reg32;
signal todelayena	 : std_logic;
signal todelayeoe	 : std_logic;

signal prescale_cnt		: STD_LOGIC_VECTOR(31 downto 0);
signal prescale_ena_r	: std_logic;
signal prescale_r			: STD_LOGIC_VECTOR(31 downto 0);

--signal chip_marker_r		: chipmarkertype;

begin


singlero: work.singlechip_ro 
	generic map(
		COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
		HITSIZE				=> HITSIZE
	)
	port map(
		reset_n				=> reset_n,
		clk					=> clk,
		counter125			=> counter125,
		link_flag			=> link_flag,
		hit_in				=> hit_in,
		hit_ena				=> hit_ena,
		coarsecounter		=> coarsecounter,
		coarsecounter_ena	=> coarsecounter_ena,
		chip_marker			=> chip_marker,		
		tomemdata			=> todelaydata,
		tomemena				=> todelayena,
		tomemeoe				=> todelayeoe
		);

process(clk)
begin
if(rising_edge(clk))then
		prescale_r		<=	prescale;
--		chip_marker_r	<= chip_marker;
	if(prescale_r = 0)then		-- enable prescaler if non-zero prescaling value is used, otherwise don't use it
		prescale_ena_r <= '0';
	else
		prescale_ena_r	<= '1';
	end if;
end if;
end process;


-- new state machine uses the same singlechip RO mode
-- checks if hits are present from the datastream
-- and drops if prescaler says so!
-- we know that there are hits, if we see 7 words in a row, otherwise there are no hits!
-- empty block consists of 7 words only!
process(reset_n, clk)
begin
if(reset_n = '0') then
	eventcounter	<= (others => '0');
--	hitspresent		<= '0';
	prescale_cnt	<= (others => '0');
	count_ena		<= (others => '0');
	delayarray		<= (others => (others => '0'));
	enadelay			<= (others => '0');
	eoedelay			<= (others => '0');
elsif(clk'event and clk = '1') then
	
	tomemena		<= '0';
	tomemeoe		<= '0';

	if(todelayena = '1')then
		delayarray(0)		<= todelaydata;
		delayarray(NDELAY-1 downto 1)	<= delayarray(NDELAY-2 downto 0);
		enadelay(0)			<= todelayena;
		enadelay(NDELAY-1 downto 1)	<= enadelay(NDELAY-2 downto 0);
		eoedelay(0)			<= todelayeoe;
		eoedelay(NDELAY-1 downto 1)	<= eoedelay(NDELAY-2 downto 0);
		tomemdata			<= delayarray(NDELAY-1);
		tomemena				<= enadelay(NDELAY-1);
		tomemeoe				<= eoedelay(NDELAY-1);
	end if;
	
	-- we also want to add the information of how many blocks were sent out actually!
	-- we put this into the "overflow state", which should be at position number 4!
	if(todelayena = '1')then				-- data coming from SingleRO
		if(count_ena < x"F") then			
			count_ena <= count_ena + 1;	-- count number of words sent, counht is only relevant up to 8
		end if;	
		if(count_ena = x"6" and todelayeoe ='0')then		-- this block contains hits, let it pass!
				delayarray(2) <= '0' & eventcounter(30 downto 0);	-- # of blocks actually sent after zero suppression
				eventcounter	<= eventcounter + 1;				
		end if;
		if(todelayeoe = '1')then			-- we see the end of event flag
			count_ena <= (others => '0');	-- and clear the counter for the next block
			if(count_ena < x"7")then	-- if the block does only contain 6 (with EOE 7) words and we don't want to send empty blocks
				if(prescale_ena_r = '0')then				-- and we don't want to send empty words:
						enadelay		<= (others => '0');	-- --> drop them!
						eoedelay		<= (others => '0');		
				elsif(prescale_cnt < prescale_r)then	-- we want to send empty words, but reduce the number by a factor of PRESCALE_R
						enadelay		<= (others => '0');	-- --> drop'em!
						eoedelay		<= (others => '0');
						prescale_cnt 	<= prescale_cnt + 1;	-- and increase the prescaling counter!
				else
						prescale_cnt 	<= (others => '0');	-- we want to send empty words + this one should be sent out, so let it go!
						delayarray(2) 	<= '0' & eventcounter(30 downto 0);	-- # of blocks actually sent after zero suppression
						eventcounter	<= eventcounter + 1;				
				end if;
				
			end if;
		end if;
	end if;
	
end if;
end process;

end rtl;