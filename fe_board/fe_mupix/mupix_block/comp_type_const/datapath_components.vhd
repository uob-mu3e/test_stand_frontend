library ieee;
use ieee.std_logic_1164.all;

use work.mupix_constants.all;
use work.mupix_types.all;
use work.mupix_registers.all;

package datapath_components is

-- telescope components  
component data_path is
generic(
	NCHIPS 			: integer :=  4;
	NGX				: integer := 14;
	NLVDS 			: integer := 16;
	NSORTERINPUTS	: integer :=  4
);
port (
	resets_n:				in reg32;
	resets: 					in reg32;
	slowclk:					in std_logic;
	clk125:					in std_logic;
	counter125:				in reg64;
	
	serial_data_in:		in std_logic_vector(NGX-1 downto 0);
	lvds_data_in:			in std_logic_vector(NLVDS-1 downto 0);
	
	clkext_out:				out std_logic_vector(NCHIPS-1 downto 0);
	
	writeregs:				in		reg32array;
	regwritten:				in 	std_logic_vector(NREGISTERS-1 downto 0);

	readregs_slow: 		out reg32array;
	
	readmem_clk:			out std_logic;
	readmem_data:			out reg32;
	readmem_addr:			out readmemaddrtype;
	readmem_wren:			out std_logic;
	readmem_eoe:			out std_logic;
	
	-- trigger interface
	readtrigfifo:			out std_logic;
	fromtrigfifo:			in reg64;
	trigfifoempty:			in std_logic;
	
	readhitbusfifo:		out std_logic;
	fromhitbusfifo:		in reg64;
	hitbusfifoempty:		in std_logic
	
--	flash_tcm_address_out                       : out   std_logic_vector(25 downto 0);                    -- tcm_address_out
--	flash_tcm_outputenable_n_out                : out   std_logic_vector(0 downto 0);                     -- tcm_read_n_out
--	flash_tcm_write_n_out                       : out   std_logic_vector(0 downto 0);                     -- tcm_write_n_out
--	flash_tcm_data_out                          : inout std_logic_vector(15 downto 0) := (others => 'X'); -- tcm_data_out
--	flash_tcm_chipselect_n_out                  : out   std_logic_vector(0 downto 0)                     -- tcm_chipselect_n_out			
	
);
end component;


component data_path_frontend is
generic(
	NCHIPS 			: integer :=  8;
	NGX				: integer :=  8;
	NLVDS 			: integer := 32;
	NSORTERINPUTS	: integer :=  4	--up to 4 LVDS links merge to one sorter
);
port (
	resets_n:				in reg32;
	resets: 					in reg32;
	slowclk:					in std_logic;
	clk125:					in std_logic;
	counter125:				in reg64;
	
	serial_data_in:		in std_logic_vector(NGX-1 downto 0);
	serial_data_out:		out std_logic_vector(NGX-1 downto 0);
		
	in_parallel_data: 	in std_logic_vector(NGX*40-1 downto 0);	-- full channel if needed
	in_k:						in std_logic_vector(NGX*5-1 downto 0);		-- full channel if needed
	out_parallel_data: 	out std_logic_vector(NGX*8-1 downto 0);	-- only Most significant bytes from RX
	out_k:					out std_logic_vector(NGX*1-1 downto 0);	-- only Most significant bytes from RX
	out_datavalid:			out std_logic_vector(NGX-1 downto 0);		-- only Most significant bytes from RX
	
	writeregs:				in	reg32array;
	regwritten:				in std_logic_vector(NREGISTERS-1 downto 0);

	readregs_slow: 		out reg32array;
	
	readmem_clk:			out std_logic;
	readmem_data:			out reg32;
	readmem_addr:			out readmemaddrtype;
	readmem_wren:			out std_logic;
	readmem_eoe:			out std_logic;
	
	-- trigger interface
	readtrigfifo:			out std_logic;
	fromtrigfifo:			in reg64;
	trigfifoempty:			in std_logic;
	
	readhitbusfifo:		out std_logic;
	fromhitbusfifo:		in reg64;
	hitbusfifoempty:		in std_logic
	
);
end component;

component memorymux IS
	PORT
	(
		clock		: IN STD_LOGIC ;
		data0x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data1x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data2x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data3x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data4x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data5x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data6x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		data7x		: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		sel		: IN STD_LOGIC_VECTOR (2 DOWNTO 0);
		result		: OUT STD_LOGIC_VECTOR (31 DOWNTO 0)
	);
END component;

component enablemux IS
	PORT
	(
		clock		: IN STD_LOGIC ;
		data0		: IN STD_LOGIC ;
		data1		: IN STD_LOGIC ;
		data2		: IN STD_LOGIC ;
		data3		: IN STD_LOGIC ;
		data4		: IN STD_LOGIC ;
		data5		: IN STD_LOGIC ;
		data6		: IN STD_LOGIC ;
		data7		: IN STD_LOGIC ;
		sel		: IN STD_LOGIC_VECTOR (2 DOWNTO 0);
		result		: OUT STD_LOGIC 
	);
END component;

component syncfifos is
generic(
	NCHIPS : integer := 4
);
port (
	ready:					in std_logic_vector(NCHIPS-1 downto 0); 
	clkout:					in std_logic; -- clock for outputs
	reset_n:					in std_logic;
		
	clkin:					in std_logic_vector(NCHIPS-1 downto 0);
	datain:					in std_logic_vector(8*NCHIPS-1 downto 0);
	kin:						in std_logic_vector(NCHIPS-1 downto 0);
	
	dataout:					out std_logic_vector(8*NCHIPS-1 downto 0);
	kout:						out std_logic_vector(NCHIPS-1 downto 0);
	
	data_valid:				out std_logic_vector(NCHIPS-1 downto 0)
--	fifo_underflow:		out chips_reg32;
--	fifo_overflow:			out chips_reg32;
--	fifo_rdusedw_out:		out reg32

);
end component;

component syncfifo
	PORT
	(
		aclr		: IN STD_LOGIC  := '0';	
		data		: IN STD_LOGIC_VECTOR (8 DOWNTO 0);
		rdclk		: IN STD_LOGIC ;
		rdreq		: IN STD_LOGIC ;
		wrclk		: IN STD_LOGIC ;
		wrreq		: IN STD_LOGIC ;
		q		: OUT STD_LOGIC_VECTOR (8 DOWNTO 0);
		rdusedw		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		wrusedw		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0)		
	);
end component;

component pseudo_data_generator_mp8 is 
	generic (
		NumCOL				: NumCOL_array	:= (32,48,48);	-- A,B: 48, C: 32
		NumROW				: integer	:= 200;				-- correct
		MatrixSEL			: MatrixSEL_array := ("00","01","10")
	);
	port (
		reset_n				: in std_logic;
		syncres				: in std_logic;
		reset_pll			: in std_logic;
		clk125				: in std_logic;			-- 125 MHz
		
		slowdown				: in std_logic_vector(27 downto 0);
		numhits				: in std_logic_vector(5 downto 0);
		
		linken				: in std_logic_vector(3 downto 0);
		
		-- readout state machine in digital part
		ckdivend				: in std_logic_vector(5 downto 0);
		ckdivend2			: in std_logic_vector(5 downto 0);
		tsphase				: in std_logic_vector(5 downto 0);
		timerend				: in std_logic_vector(3 downto 0);
		slowdownend			: in std_logic_vector(3 downto 0);
		maxcycend			: in std_logic_vector(5 downto 0);	
		resetckdivend		: in std_logic_vector(3 downto 0);	
		sendcounter			: in std_logic;
		linksel				: in std_logic_vector(1 downto 0);
		mode					: in std_logic_vector(1 downto 0);
		
		dataout				: out std_logic_vector(31 downto 0);
		kout					: out std_logic_vector(3 downto 0);
		syncout				: out std_logic_vector(3 downto 0);

		state_out			: out std_logic_vector(23 downto 0)
		);
end component;

component data_unpacker_triple is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		UNPACKER_HITSIZE	: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
--		reset_out_n			: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		is_shared			: IN STD_LOGIC;
		timerend				: IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		hit_out				: OUT STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);		-- Link[7:0] & Row[7:0] & Col[7:0] & Charge[5:0] & TS[9:0]
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);	-- Gray Counter[7:0] & Binary Counter [23:0]
		coarsecounter_ena	: OUT STD_LOGIC;
		link_flag			: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0);
		error_out			: OUT STD_LOGIC_VECTOR(31 downto 0);
		readycounter		: OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
		);
end component;

component data_unpacker_triple_new is
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		UNPACKER_HITSIZE	: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
--		reset_out_n			: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		is_shared			: IN STD_LOGIC;
		is_atlaspix			: IN STD_LOGIC;
--		timerend				: IN STD_LOGIC_VECTOR(3 DOWNTO 0);
		hit_out				: OUT STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);		-- Link[7:0] & Row[7:0] & Col[7:0] & Charge[5:0] & TS[9:0]
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);	-- Gray Counter[7:0] & Binary Counter [23:0]
		coarsecounter_ena	: OUT STD_LOGIC;
		link_flag			: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0)
--		error_out			: OUT STD_LOGIC_VECTOR(31 downto 0);
--		readycounter		: OUT STD_LOGIC_VECTOR(31 DOWNTO 0)
		);
end component;

component data_unpacker is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		UNPACKER_HITSIZE	: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
--		reset_out_n			: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		mux_ready			: IN STD_LOGIC;
		hit_out				: OUT STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounter_ena	: OUT STD_LOGIC;
		link_flag			: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0);
		error_out			: OUT STD_LOGIC_VECTOR(31 downto 0)		
		);
end component;

component data_unpacker_new is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		UNPACKER_HITSIZE	: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		is_atlaspix			: IN STD_LOGIC;
		hit_out				: OUT STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);		-- Link[7:0] & Row[7:0] & Col[7:0] & Charge[5:0] & TS[9:0]
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);	-- Gray Counter[7:0] & Binary Counter [23:0]
		coarsecounter_ena	: OUT STD_LOGIC;
		link_flag			: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0)
		);
end component;

component data_unpacker_mp7 is
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= 40
	);
	port (
		reset_n				: in std_logic;
--		reset_out_n			: in std_logic;
		clk					: in std_logic;
		datain				: IN STD_LOGIC_VECTOR (7 DOWNTO 0);
		kin					: IN STD_LOGIC;
		readyin				: IN STD_LOGIC;
		hit_out				: OUT STD_LOGIC_VECTOR (HITSIZE-1 DOWNTO 0);
		hit_ena				: OUT STD_LOGIC;
		coarsecounter		: OUT STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounter_ena	: OUT STD_LOGIC;
		errorcounter		: OUT STD_LOGIC_VECTOR(31 downto 0);
		error_out			: OUT STD_LOGIC_VECTOR(31 downto 0)
		);
end component;

component hit_gray_to_binary is 
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_in			: in STD_LOGIC;
		hit_out				: out STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_out			: out STD_LOGIC
		);
end component;

component hit_gray_to_binary_TOT is 
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_in			: in STD_LOGIC;
		hit_out				: out STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_out			: out STD_LOGIC
		);
end component;

component gray_to_binary is 
	generic(NBITS : integer :=10);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		gray_in				: in std_logic_vector (NBITS-1 DOWNTO 0);
		bin_out				: out std_logic_vector (NBITS-1 DOWNTO 0)
		);
end component;

component hit_ts_conversion is 
	generic(TS_SIZE 			: integer :=  TIMESTAMPSIZE_MPX8);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		invert_TS			: in std_logic;
		invert_TS2			: in std_logic;
		gray_TS				: in std_logic;
		gray_TS2				: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_in			: in STD_LOGIC;
		hit_out				: out STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena_out			: out STD_LOGIC
		);
end component;

component binary_to_gray is 
	generic(NBITS : integer :=10);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		bin_in				: in std_logic_vector (NBITS-1 DOWNTO 0);
		gray_out				: out std_logic_vector (NBITS-1 DOWNTO 0)
		);
end component;

component singlemux IS
	PORT
	(
		clock		: IN STD_LOGIC ;
		data0x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data10x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data11x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data12x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data13x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data14x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data15x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data1x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data2x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data3x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data4x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data5x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data6x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data7x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data8x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		data9x		: IN STD_LOGIC_VECTOR (74 DOWNTO 0);
		sel			: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		result		: OUT STD_LOGIC_VECTOR (74 DOWNTO 0)
	);
END component;

component singlechip_ro is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= UNPACKER_HITSIZE
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		counter125			: in reg64;
		link_flag			: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena				: in STD_LOGIC;
		coarsecounter		: in STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounter_ena	: in STD_LOGIC;
		chip_marker			: in chipmarkertype;		
		tomemdata			: out reg32;
		tomemena				: out std_logic;
		tomemeoe				: out std_logic
		);
end component;

component singlechip_ro_zerosupressed is 
	generic (
		COARSECOUNTERSIZE	: integer	:= 32;
		HITSIZE				: integer	:= UNPACKER_HITSIZE
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		counter125			: in reg64;
		link_flag			: in std_logic;
		hit_in				: in STD_LOGIC_VECTOR (UNPACKER_HITSIZE-1 DOWNTO 0);
		hit_ena				: in STD_LOGIC;
		coarsecounter		: in STD_LOGIC_VECTOR (COARSECOUNTERSIZE-1 DOWNTO 0);
		coarsecounter_ena	: in STD_LOGIC;
		chip_marker			: in chipmarkertype;	
		prescale				: in STD_LOGIC_VECTOR(31 downto 0);
		tomemdata			: out reg32;
		tomemena				: out std_logic;
		tomemeoe				: out std_logic
		);
end component;

component hit_serializer is 
	generic(
	SERIALIZER_HITSIZE : integer := UNPACKER_HITSIZE+4;
	SERBINCOUNTERSIZE	: integer := BINCOUNTERSIZE+4;
	SERHITSIZE : integer := 40
	);
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		hit_in1				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena1				: IN STD_LOGIC;
		hit_in2				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena2				: IN STD_LOGIC;
		hit_in3				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena3				: IN STD_LOGIC;
		hit_in4				: IN STD_LOGIC_VECTOR (SERIALIZER_HITSIZE-1 DOWNTO 0);
		hit_ena4				: IN STD_LOGIC;
		time_in1				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena1			: IN STD_LOGIC;
		time_in2				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena2			: IN STD_LOGIC;
		time_in3				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena3			: IN STD_LOGIC;
		time_in4				: IN STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena4			: IN STD_LOGIC;
		hit_out				: OUT STD_LOGIC_VECTOR (SERHITSIZE-1 DOWNTO 0);
		hit_ena				: OUT STD_LOGIC;
		time_out				: OUT STD_LOGIC_VECTOR (SERBINCOUNTERSIZE-1 DOWNTO 0);
		time_ena				: OUT STD_LOGIC
		);
end component;

component hit_counter is
	PORT( 
		clock 		: 	in std_logic;
		reset_n 		: 	in std_logic;
		coarse_ena	: 	in std_logic;		
		hits_in		:	in std_logic_vector(MHITSIZE-1 downto 0);
		hits_ena_in	: 	in std_logic;
		counter		:	out reg48array;
		counter_sum	: 	out std_logic_vector(47 downto 0)
	);
end component;

component single_zs_fifo IS
	PORT
	(
		aclr		: IN STD_LOGIC;
		clock		: IN STD_LOGIC ;
		data		: IN STD_LOGIC_VECTOR (32 DOWNTO 0);
		rdreq		: IN STD_LOGIC ;
		wrreq		: IN STD_LOGIC ;
		empty		: OUT STD_LOGIC ;
		full		: OUT STD_LOGIC ;
		q			: OUT STD_LOGIC_VECTOR (32 DOWNTO 0);
		usedw		: OUT STD_LOGIC_VECTOR (8 DOWNTO 0)
	);
END component;

component timemuxed_ro_zerosupressed is 
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
end component;

component sorter_debugmux IS
	PORT
	(
		data0x		: IN STD_LOGIC_VECTOR (255 DOWNTO 0);
		data1x		: IN STD_LOGIC_VECTOR (255 DOWNTO 0);
		data2x		: IN STD_LOGIC_VECTOR (255 DOWNTO 0);
		data3x		: IN STD_LOGIC_VECTOR (255 DOWNTO 0);
		sel		: IN STD_LOGIC_VECTOR (1 DOWNTO 0);
		result		: OUT STD_LOGIC_VECTOR (255 DOWNTO 0)
	);
END component;

component hitsorter_debug_mux is 
	port (
		clk					: in std_logic;
		dataA0x				: in reg32;
		dataA1x				: in reg32;
		dataA2x				: in reg32;
		dataA3x				: in reg32;
		dataA4x				: in reg32;
		dataA5x				: in reg32;
		dataA6x				: in reg32;
		dataA7x				: in reg32;
		dataB0x				: in reg32;
		dataB1x				: in reg32;
		dataB2x				: in reg32;
		dataB3x				: in reg32;
		dataB4x				: in reg32;
		dataB5x				: in reg32;
		dataB6x				: in reg32;
		dataB7x				: in reg32;
		dataC0x				: in reg32;
		dataC1x				: in reg32;
		dataC2x				: in reg32;
		dataC3x				: in reg32;
		dataC4x				: in reg32;
		dataC5x				: in reg32;
		dataC6x				: in reg32;
		dataC7x				: in reg32;
		dataD0x				: in reg32;
		dataD1x				: in reg32;
		dataD2x				: in reg32;
		dataD3x				: in reg32;
		dataD4x				: in reg32;
		dataD5x				: in reg32;
		dataD6x				: in reg32;
		dataD7x				: in reg32;
		sel 					: in std_logic_vector(1 downto 0);
		dout0					: out reg32;		
		dout1					: out reg32;
		dout2					: out reg32;
		dout3					: out reg32;
		dout4					: out reg32;
		dout5					: out reg32;
		dout6					: out reg32;
		dout7					: out reg32
		);
end component;


-- standalone components 
component data_path_standalone is
generic(
	NCHIPS : integer := 2
);
port (
	-- resets and clocking
	resets_n:				in reg32;
	slowclk:					in std_logic;
	clk125:					in std_logic;
	-- Timestamp
	counter125:				in reg64;
	-- serial data input for fast GX receivers
	serial_data_in:		in std_logic_vector(NCHIPS-1 downto 0);
	-- registers to control data path
	writeregs:				in		reg32array;
	regwritten:				in 	std_logic_vector(NREGISTERS-1 downto 0);
	-- registers to monitor data path
	readregs_slow: 		out reg32array;
	-- memory writer
	readmem_clk:			out std_logic;
	readmem_data:			out reg32;
	readmem_addr:			out readmemaddrtype;
	readmem_wren:			out std_logic;
	readmem_eoe:			out std_logic;
	-- trigger interface
	readtrigfifo:			out std_logic;
	fromtrigfifo:			in reg64;
	trigfifoempty:			in std_logic;
	-- hitbus interface	
	readhitbusfifo:			out std_logic;
	fromhitbusfifo:			in reg64;
	hitbusfifoempty:			in std_logic;
	tx_debug_serialout	: OUT STD_LOGIC_VECTOR(1 downto 0)
	
);
end component;

component hit_counter_simple is
	PORT( 
		clock 		: 	in std_logic;
		reset_n 		: 	in std_logic;
		coarse_ena	: 	in std_logic;
		hits_ena_in	: 	in std_logic;
		counter		:	out std_logic_vector(47 downto 0)
	);
end component;

  
end package datapath_components;
