-- Constants solely for use in mupix block
-- check which of these are actually used and remove those that are not needed!

library ieee;
use ieee.std_logic_1164.all;

package mupix_constants is
-----------------------------------------------------------------
-- conflicts between detectorfpga_constants and mupix_constants
-----------------------------------------------------------------

--constant NCHIPS                             : integer :=  8;
constant NCHIPS 	: integer := 12;

-- constant HITSIZE                            : integer := 24;
constant HITSIZE 	: integer := 32;

constant TIMESTAMPSIZE_MP10                 : integer := 11;
constant TIMESTAMPSIZE		: integer := 11;

subtype TSRANGE is integer range TIMESTAMPSIZE-1 downto 0;
--subtype  TSRANGE            is integer range 9 downto 0;

constant MHITSIZE	: integer := HITSIZE+2;
--constant MHITSIZE                           : integer := 42;--26; -- Merge hits from four chips

constant COARSECOUNTERSIZE : integer := 32;
--constant COARSECOUNTERSIZE                  : integer := 32;

--subtype COLRANGE is integer range 31 downto 24;
--subtype ROWRANGE is integer range 23 downto 16;
subtype  COLRANGE           is integer range 23 downto 16;
subtype  ROWRANGE           is integer range 15 downto 8;

--constant CHIPRANGE			: integer := 2;
constant CHIPRANGE                          : integer := 3;

-----------------------------------------------------------
-- stuff from old mupix_constants (to be cleaned up)
-----------------------------------------------------------

-- read register number
constant NREGISTERS_MUPIX_RD                : integer := 94;
-- read register map
constant RX_STATE_RECEIVER_0_REGISTER_R     : integer := 16#00#;    -- dec 0
constant RX_STATE_RECEIVER_1_REGISTER_R     : integer := 16#01#;    -- dec 1
constant LVDS_PLL_LOCKED_REGISTER_R         : integer := 16#02#;    -- dec 2
constant MULTICHIP_RO_OVERFLOW_REGISTER_R   : integer := 16#03#;    -- dec 3
constant LVDS_RUNCOUNTER_REGISTER_R         : integer := 16#04#;    -- dec 4 (to 48)
constant LVDS_ERRCOUNTER_REGISTER_R         : integer := 16#30#;    -- dec 49 (to 93)

-- write register number
constant NREGISTERS_MUPIX_WR                : integer := 5;
-- read register map
constant RO_PRESCALER_REGISTER_W            : integer := 16#00#;    -- dec 0
constant DEBUG_CHIP_SELECT_REGISTER_W       : integer := 16#01#;
constant TIMESTAMP_GRAY_INVERT_REGISTER_W   : integer := 16#02#;
constant TS_INVERT_BIT                      : integer := 0;
constant TS2_INVERT_BIT                     : integer := 1;
constant TS_GRAY_BIT                        : integer := 2;
constant TS2_GRAY_BIT                       : integer := 3;
constant LINK_MASK_REGISTER_W               : integer := 16#03#;

-- constants
constant NFEB_CARDS                         : integer :=  4;
constant NINJECTIONS                        : integer :=  1;
constant NSORTERINPUTS                      : integer :=  1;    -- 8 does not fit into FPGA :(

constant NMATRIX                            : integer := 3; -- for pseudo data generator

constant BINCOUNTERSIZE                     : integer := 24;
constant UNPACKER_HITSIZE                   : integer := 32;
subtype  BINCOUNTERRANGE    is integer range 23 downto 0;
subtype  TBLOCKRANGEplus1   is integer range 8 downto 6;
subtype  TSRANGEplus1       is integer range 8 downto 1;
subtype  TBLOCKRANGE        is integer range 7 downto 5;
constant INVALID                            : STD_LOGIC_VECTOR(7 downto 0) := x"00";
constant CHARGESIZE_MP10                    : integer := 5;

constant CHIPMARKERSIZE                     : integer := 8;

constant BEGINOFEVENT                       : std_logic_vector(31 downto 0) := x"FABEABBA";
constant ENDOFEVENT                         : std_logic_vector(31 downto 0) := x"BEEFBEEF";
constant HITMARKER                          : std_logic_vector(5 downto 0) := "101010";
constant HITMARKERSIZE                      : integer := 6;

constant BEGINOFTRIGGER                     : std_logic_vector(31 downto 0) := x"CAFECAFE";
constant ENDOFTRIGGER                       : std_logic_vector(31 downto 0) := x"CAFEBABE";

constant BEGINOFHB                          : std_logic_vector(31 downto 0) := x"BADEBADE";
constant ENDOFHB                            : std_logic_vector(31 downto 0) := x"BEADDEED";

constant DDR3_OVERFLOW                      : std_logic_vector(31 downto 0) := x"800BB008";

constant DATAFAKER_SEED                     : integer := 16;

constant HITLABEL                           : std_logic_vector(3 downto 0) := x"E";
constant TIMELABEL                          : std_logic_vector(3 downto 0) := x"F";

-----------------------------------------------------------
-- stuff from detectorfpga_constants (to be cleaned up)
-----------------------------------------------------------

constant NINPUTS 	: integer := 36;
constant NLADDERS 	: integer := 4;
constant SLOWTIMESTAMPSIZE	: integer := 10;
constant NTIMESTAMPS	: integer := 2**TIMESTAMPSIZE;
constant NSLOWTIMESTAMPS	: integer := 2**SLOWTIMESTAMPSIZE;
constant NOTSHITSIZE	: integer := HITSIZE -TIMESTAMPSIZE-1;
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

end package mupix_constants;
