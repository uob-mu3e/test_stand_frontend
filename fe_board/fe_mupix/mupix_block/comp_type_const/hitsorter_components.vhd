-- Components, types and constants for the MuPix8-hitsorter

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;
use work.mupix_types.all;


package hitsorter_components is

-- CONSTANTS
constant HITSIZE: integer := 2+2+8+8+6+10;	-- serializer + link + Row + Col + Charge + TS
constant MEMHITSIZE:	integer := 40;
constant NINPUT: integer:= NSORTERINPUTS;

constant DUMMY	: std_logic_vector(31 downto 0)	:= x"DEADDA7A";

-- TYPES
subtype hit_t is std_logic_vector(MEMHITSIZE-1 downto 0);
type hit_array_t is array (NINPUT-1 downto 0) of hit_t;
subtype counter_t is std_logic_vector(BINCOUNTERSIZE+4-1 downto 0);
type counter_array_t is array (NINPUT-1 downto 0) of counter_t;

--subtype TSRANGE is integer range 9 downto 0; -- already in mupix_constants

--COMPONENTS
COMPONENT hitsortermem IS
	PORT
	(
		data			: IN STD_LOGIC_VECTOR (29 DOWNTO 0);
		rdaddress	: IN STD_LOGIC_VECTOR (12 DOWNTO 0);
		rdclock		: IN STD_LOGIC ;
		wraddress	: IN STD_LOGIC_VECTOR (12 DOWNTO 0);
		wrclock		: IN STD_LOGIC  := '1';
		wren			: IN STD_LOGIC  := '0';
		q				: OUT STD_LOGIC_VECTOR (29 DOWNTO 0)
	);
END component;

component hitsorter is
	generic(
		NINPUT:	integer;	-- 1 FPGA has 16 inputs, i.e. 4 inputs to the sorter with a 4to1Mux before the sorter
		NSORTERINPUTS:	integer
--		NINPUT2:	integer := 1; -- half NINPUT or 1 (whichever is bigger)
--		NINPUT4:	integer := 1; -- one quarter NINPUT or 1 (whichever is bigger)
--		NINPUT8:  integer := 1; -- one eight NINPUT or 1 (whichever is bigger)
--		NINPUT16:  integer := 1;-- one sixteenth NINPUT or 1 (whichever is bigger)
--		HITSIZE: integer := 2+2+8+8+6+10	-- serializer + link + Row + Col + Charge + TS
--		
	);
	port (
		reset_n	:	in	std_logic;
		writeclk: 	in	std_logic;
		tsdivider:	in std_logic_vector(3 downto 0);
		readclk:		in	std_logic;
		
		hits_ser:			in hit_array_t;	--4x40b
		hits_ser_en:		in std_logic_vector(NINPUT-1 downto 0);
		ts_en:		in std_logic_vector(NINPUT-1 downto 0);
		
		tsdelay:		in	std_logic_vector(TSRANGE);
		
		fromtrigfifo: 	in reg64;
		trigfifoempty:	in std_logic;	
		fromhbfifo: 	in reg64;
		hbfifoempty: 	in std_logic;	
		
		data_out:	out std_logic_vector(31 downto 0);
		out_ena:		out std_logic;
		out_eoe:		out std_logic;
		
		out_hit:		out std_logic;
		
		readtrigfifo:	out std_logic;
		readhbfifo: 	out std_logic;	
		
		received_hits:		out std_logic_vector(31 downto 0);	
		outoftime_hits:	out std_logic_vector(31 downto 0);	
		intime_hits:		out std_logic_vector(31 downto 0);	
		memwrite_hits:		out std_logic_vector(31 downto 0);	
		overflow_hits:		out std_logic_vector(31 downto 0);	
		sent_hits:			out std_logic_vector(31 downto 0);
		break_counter:		out std_logic_vector(31 downto 0)			
	);
end component;




end package;
