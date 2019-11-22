-- Constants solely for use in mupix block

library ieee;
use ieee.std_logic_1164.all;

package mupix_constants is

-- read register number
constant NREGISTERS_MUPIX_RD:	integer := 68;
-- read register map
constant RX_STATE_RECEIVER_0_REGISTER_R		: integer := 16#00#;	-- dec 0
constant RX_STATE_RECEIVER_1_REGISTER_R		: integer := 16#01#;	-- dec 1
constant LVDS_PLL_LOCKED_REGISTER_R				: integer := 16#02#;	-- dec 2
constant MULTICHIP_RO_OVERFLOW_REGISTER_R		: integer := 16#03#;	-- dec 3
constant LVDS_RUNCOUNTER_REGISTER_R	 			: integer := 16#04#;	-- dec 4 (to 35)
constant LVDS_ERRCOUNTER_REGISTER_R				: integer := 16#24#;	-- dec 36 (to 67)

-- write register number
constant NREGISTERS_MUPIX_WR:	integer := 4;
-- read register map
constant RO_PRESCALER_REGISTER_W							: integer := 16#00#;	-- dec 0
constant DEBUG_CHIP_SELECT_REGISTER_W					: integer := 16#01#;
constant TIMESTAMP_GRAY_INVERT_REGISTER_W				: integer := 16#02#;
	constant TS_INVERT_BIT	: integer := 0;
	constant TS2_INVERT_BIT	: integer := 1;
	constant TS_GRAY_BIT		: integer := 2;
	constant TS2_GRAY_BIT	: integer := 3;
constant LINK_MASK_REGISTER_W								: integer := 16#03#;

-- constants
constant NCHIPS			: integer :=  8;
constant NFEB_CARDS		: integer :=  4;
constant NINJECTIONS		: integer :=  1;
constant NSORTERINPUTS	: integer :=  1;	-- 8 does not fit into FPGA :(

constant NMATRIX		: integer := 3; -- for pseudo data generator

constant COARSECOUNTERSIZE : integer := 32;
constant BINCOUNTERSIZE		: integer := 24;
constant HITSIZE				: integer := 24;
constant UNPACKER_HITSIZE	: integer := 40;
subtype 	BINCOUNTERRANGE	is integer range 23 downto 0;
subtype COLRANGE is integer range 23 downto 16;
subtype ROWRANGE is integer range 15 downto 8;
subtype TSRANGE is integer range 9 downto 0;
subtype TBLOCKRANGEplus1 is integer range 8 downto 6;
subtype TSRANGEplus1 is integer range 8 downto 1;
subtype TBLOCKRANGE is integer range 7 downto 5;
constant INVALID 				: STD_LOGIC_VECTOR(7 downto 0) := x"00";
constant MHITSIZE				: integer := 42;--26; -- Merge hits from four chips
constant TIMESTAMPSIZE		: integer := 8;
constant TIMESTAMPSIZE_MPX8: integer := 10;
constant CHARGESIZE			: integer := 6;

constant CHIPMARKERSIZE 	: integer := 8;
constant CHIPRANGE			: integer := 4;

constant BEGINOFEVENT 		: std_logic_vector(31 downto 0) := x"FABEABBA";
constant ENDOFEVENT 			: std_logic_vector(31 downto 0) := x"BEEFBEEF";
constant HITMARKER      	: std_logic_vector(5 downto 0) := "101010";
constant HITMARKERSIZE  	: integer := 6;

constant BEGINOFTRIGGER 	: std_logic_vector(31 downto 0) := x"CAFECAFE";
constant ENDOFTRIGGER 		: std_logic_vector(31 downto 0) := x"CAFEBABE";

constant BEGINOFHB 			: std_logic_vector(31 downto 0) := x"BADEBADE";
constant ENDOFHB 				: std_logic_vector(31 downto 0) := x"BEADDEED";

constant DDR3_OVERFLOW : std_logic_vector(31 downto 0) := x"800BB008";

constant DATAFAKER_SEED		: integer := 16;

constant HITLABEL				: std_logic_vector(3 downto 0) := x"E";
constant TIMELABEL			: std_logic_vector(3 downto 0) := x"F";



end package mupix_constants;
