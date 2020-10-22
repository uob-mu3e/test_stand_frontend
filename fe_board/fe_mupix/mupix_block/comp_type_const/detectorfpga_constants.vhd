---------------------------------------
--
-- On detector FPGA for layer 0 - constant library
-- Sebastian Dittmeier, June 2016
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;

package detectorfpga_constants is

constant NINPUTS 	: integer := 36;
constant NCHIPS 	: integer := 12;
constant NLADDERS 	: integer := 4;
constant HITSIZE 	: integer := 32;
constant TIMESTAMPSIZE		: integer := 11;
constant SLOWTIMESTAMPSIZE	: integer := 10;
constant NTIMESTAMPS	: integer := 2**TIMESTAMPSIZE;
constant NSLOWTIMESTAMPS	: integer := 2**SLOWTIMESTAMPSIZE;
constant NOTSHITSIZE	: integer := HITSIZE -TIMESTAMPSIZE-1;
subtype TSRANGE is integer range TIMESTAMPSIZE-1 downto 0;
subtype SLOWTSRANGE is integer range TIMESTAMPSIZE-1 downto 1;
subtype NOTSRANGE is integer range HITSIZE-1 downto TIMESTAMPSIZE+1;

constant HITSORTERBINBITS : integer := 4;
constant H	: integer := HITSORTERBINBITS;
constant HITSORTERNADDR	  : integer := 2**HITSORTERBINBITS;
constant HITSORTERADDRSIZE: integer := TIMESTAMPSIZE + HITSORTERBINBITS;

constant BITSPERTSBLOCK : integer := 4;
subtype TSBLOCKRANGE is integer range TIMESTAMPSIZE-1 downto BITSPERTSBLOCK;
subtype SLOWTSBLOCKRANGE is integer range TIMESTAMPSIZE-2 downto BITSPERTSBLOCK-1;
subtype TSNONBLOCKRANGE is integer range BITSPERTSBLOCK-1 downto 0;
subtype SLOWTSNONBLOCKRANGE is integer range BITSPERTSBLOCK-2 downto 0;
constant NUMTSBLOCKS : integer := 2**(TIMESTAMPSIZE-BITSPERTSBLOCK);
constant TSPERBLOCK : integer := 2**BITSPERTSBLOCK;
constant TSBLOCKBITS : integer := TIMESTAMPSIZE - BITSPERTSBLOCK;

constant COMMANDBITS : integer := 20;

constant COUNTERMEMADDRSIZE : integer := 8;
constant COUNTERMEMSELBITS : integer := TIMESTAMPSIZE-COUNTERMEMADDRSIZE-1;
constant NMEMS : integer := 2**(TIMESTAMPSIZE-COUNTERMEMADDRSIZE-1); -- -1 due to even odd in single memory
constant COUNTERMEMDATASIZE : integer := 10;
subtype COUNTERMEMSELRANGE is integer range TIMESTAMPSIZE-1 downto COUNTERMEMADDRSIZE+1;
subtype SLOWTSCOUNTERMEMSELRANGE is integer range TIMESTAMPSIZE-2 downto COUNTERMEMADDRSIZE;
subtype COUNTERMEMADDRRANGE is integer range COUNTERMEMADDRSIZE downto 1;
subtype SLOWCOUNTERMEMADDRRANGE is integer range COUNTERMEMADDRSIZE-1 downto 0;

-- Bit positions in the counter fifo of the sorter
subtype EVENCOUNTERRANGE is integer range 2*NCHIPS*HITSORTERBINBITS-1 downto 0;
constant EVENOVERFLOWBIT : integer := 2*NCHIPS*HITSORTERBINBITS;
constant HASEVENBIT : integer := 2*NCHIPS*HITSORTERBINBITS+1;
subtype ODDCOUNTERRANGE is integer range 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT downto HASEVENBIT+1;
constant ODDOVERFLOWBIT : integer := 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+1;
constant HASODDBIT : integer := 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+2;
subtype TSINFIFORANGE is integer range 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+SLOWTIMESTAMPSIZE+2 downto 2*NCHIPS*HITSORTERBINBITS+HASEVENBIT+3;
subtype TSBLOCKINFIFORANGE is integer range TSINFIFORANGE'left downto TSINFIFORANGE'left-BITSPERTSBLOCK+1;
subtype TSINBLOCKINFIFORANGE is integer range TSINFIFORANGE'right+BITSPERTSBLOCK-2  downto TSINFIFORANGE'right;

constant MHITSIZE	: integer := HITSIZE+2;

constant COARSECOUNTERSIZE : integer := 24;

constant ONE_MILLION:			std_logic_vector(19 downto 0) := x"F4240";
constant ONE_MILLION_int:		integer := 1000000;
constant ONE_MILLION32:			std_logic_vector(31 downto 0) := x"000F4240";
constant ONE_THOUSAND_int:		integer := 1000;
constant HUNDRED_MILLION: 		std_logic_vector(27 downto 0) := x"5F5E100"; 
constant HUNDRED_MILLION32:	std_logic_vector(31 downto 0) := x"05F5E100";

constant THIRTYNINE_MILLION : std_logic_vector(27 downto 0) := x"25317C0";
--constant FOURTYONE_MILLION : std_logic_vector(27 downto 0) := x"2719C40";

constant EIGHTY_MILLION : std_logic_vector(27 downto 0) := x"4C4B400";
constant EIGHTY_THOUSAND : std_logic_vector(19 downto 0) := x"13880";

constant HUNDREDTWENTYFOUR_MILLION : std_logic_vector(27 downto 0) := x"7641700";
--constant HUNDREDTWENTYSIX_MILLION : std_logic_vector(27 downto 0) := x"7829B80";

constant k28_5 : std_logic_vector(7 downto 0) := "10111100";
constant k28_1 : std_logic_vector(7 downto 0) := "00111100";

-- just from adding hitsorter
constant BINCOUNTERSIZE		: integer := 24;
 -- Merge hits from four chips
subtype 	BINCOUNTERRANGE	is integer range 23 downto 0;
subtype COLRANGE is integer range 31 downto 24;
subtype ROWRANGE is integer range 23 downto 16;
--subtype TSRANGE is integer range 10 downto 0;
--subtype TBLOCKRANGEplus1 is integer range 8 downto 6;
--subtype TSRANGEplus1 is integer range 8 downto 1;
--subtype TBLOCKRANGE is integer range 7 downto 5;
constant INVALID : STD_LOGIC_VECTOR(7 downto 0) := x"00";

constant CHIPMARKERSIZE 	: integer := 8;
constant CHIPRANGE			: integer := 2;

constant BEGINOFEVENT 	: std_logic_vector(31 downto 0) := x"FABEABBA";
constant ENDOFEVENT 		: std_logic_vector(31 downto 0) := x"BEEFBEEF";
constant HITMARKER      : std_logic_vector(5 downto 0) := "101010";
constant HITMARKERSIZE  : integer := 6;

constant BEGINOFTRIGGER : std_logic_vector(31 downto 0) := x"CAFECAFE";
constant ENDOFTRIGGER : std_logic_vector(31 downto 0) := x"CAFEBABE";

constant BEGINOFHB : std_logic_vector(31 downto 0) := x"BADEBADE";
constant ENDOFHB : std_logic_vector(31 downto 0) := x"BEADDEED";


end package detectorfpga_constants;