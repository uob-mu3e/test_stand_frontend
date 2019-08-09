-- Constants for use in mupix telescope

library ieee;
use ieee.std_logic_1164.all;

package mupix_constants is

constant NCHIPS			: integer :=  8;
constant NFEB_CARDS		: integer :=  4;
constant NTRIGGERS		: integer :=  4;
constant NREGISTERS		: integer := 64; 
constant NINJECTIONS		: integer :=  1;
constant NLVDS				: integer := 32;
-- this should be equal to log2(NLVDS)
constant NLVDSLOG			: integer := 5;
constant NGX 				: integer :=  8;
constant NSORTERINPUTS	: integer :=  2;	-- 8 does not fit into FPGA :(

constant NMATRIX		: integer := 3; -- for pseudo data generator

constant TIME_125MHz_1s 	: STD_LOGIC_VECTOR(27 DOWNTO 0) := x"7735940";
constant TIME_125MHz_1ms 	: STD_LOGIC_VECTOR(27 DOWNTO 0) := x"001E848";
constant TIME_125MHz_2s		: STD_LOGIC_VECTOR(27 DOWNTO 0) := x"EE6B280";

constant HUNDRED_MILLION: 		std_logic_vector(27 downto 0) := x"5F5E100"; 
constant HUNDRED_MILLION32:	std_logic_vector(31 downto 0) := x"05F5E100";

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


-- 8bit/10bit encoding
constant k28_0	:	std_logic_vector(7 downto 0)	:= X"1C";	-- used in MuPix
constant k28_1	:	std_logic_vector(7 downto 0)	:= X"3C";	-- used in data alignment (transceiver)
constant k28_2 : 	std_logic_vector(7 downto 0)	:= X"5C";
constant k28_3 : 	std_logic_vector(7 downto 0)	:= X"7C";
constant k28_4 : 	std_logic_vector(7 downto 0)	:= X"9C";
constant k28_5	: 	std_logic_vector(7 downto 0)	:= X"BC";	-- used in MuPix
constant k28_6 : 	std_logic_vector(7 downto 0)	:= X"DC";
constant k28_7 : 	std_logic_vector(7 downto 0)	:= X"FC";	-- not used, comma symbol with harder constraints!
constant k23_7 : 	std_logic_vector(7 downto 0)	:= X"F7";	-- used as "empty" data (transceiver)
constant k27_7 : 	std_logic_vector(7 downto 0)	:= X"FB";
constant k29_7 : 	std_logic_vector(7 downto 0)	:= X"FD";
constant k30_7 : 	std_logic_vector(7 downto 0)	:= X"FE";

end package mupix_constants;
