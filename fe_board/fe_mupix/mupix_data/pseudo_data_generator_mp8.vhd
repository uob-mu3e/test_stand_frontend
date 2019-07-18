--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
--- MuPix8 Pseudo Data Generator
--
--  Jens Kroeger
--  kroeger@physi.uni-heidelberg.de
--  April 2017
--
--  * inspired by Ann-Kathrin's pseudo_data_generator.vhd for the MuPix7
--  * includes original MuPix8 verilog code for the digital part of the chip
--
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
--  Place just before data_unpacker in datapath.
--  To be muxed with real data.
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
--use work.lfsr_pkg.all;
use work.mupix_types.all;
use work.mupix_constants.all;
use work.data_gen_components.all;


entity pseudo_data_generator_mp8 is 
	generic (
		NumCOL				: NumCOL_array	:= (32,48,48);	-- A,B: 48, C: 32
		NumROW				: integer	:= 200;				-- correct
		MatrixSEL			: MatrixSEL_array := ("00","01","10")
	);
	port (
		reset_n				: in std_logic;
		syncres				: in std_logic;			-- high-active!
		reset_pll			: in std_logic;			-- high-active!
		clk125				: in std_logic;			-- 125 MHz -- do not feed into digital part!
																			  --> goes to pll
		linken				: in std_logic_vector(3 downto 0); -- to enable links A-D
		
		-- talk to submatrices
		slowdown				: in std_logic_vector(27 downto 0); -- control frequency with which snapshots are taken
		numhits				: in std_logic_vector(5 downto 0);
		
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
end pseudo_data_generator_mp8;

architecture RTL of pseudo_data_generator_mp8 is
	
	signal pll_locked			: std_logic;
	signal reset_combined_n : std_logic;
	signal clk125_loopback 	: std_logic;
	--signal clk625			  	: std_logic;
	
	signal matrix_sel : std_logic_vector(NMATRIX*2-1 downto 0) := "100100";	--C: 10, B: 01, A: 00
	
	-- digital part to matrix
	signal ts_todet	: std_logic_vector(9 downto 0);
	signal ts_todet2	: std_logic_vector(5 downto 0);
	signal ldcol		: std_logic_vector(2 downto 0); -- for A,B,C
	signal rdcol		: std_logic_vector(2 downto 0); -- for A,B,C
	signal ldpix		: std_logic_vector(2 downto 0); -- for A,B,C
	signal pulldn		: std_logic_vector(2 downto 0); -- for A,B,C
	signal clk_tomatrix : std_logic;
	-- matrix to digital part
	signal pri_fromdet 		: std_logic_vector(NMATRIX-1 downto 0); 		-- for A,B,C
	signal rowadd_fromdet 	: std_logic_vector(NMATRIX*8-1 downto 0); 	-- for A,B,C
	signal coladd_fromdet 	: std_logic_vector(NMATRIX*8-1 downto 0); 	-- for A,B,C
	signal ts_fromdet 		: std_logic_vector(NMATRIX*16-1 downto 0); 	-- for A,B,C
	
	-- dataout registers
	signal dataout_reg : std_logic_vector(31 downto 0);
	signal kout_reg 		: std_logic_vector(3 downto 0);
	
	-- data/k to and from fifo
	signal data_tofifo	: std_logic_vector(35 downto 0);
	signal data_fromfifo : std_logic_vector(35 downto 0);
	
	signal first_shot_out 	: std_logic_vector(NMATRIX-1 downto 0) 	:= (others => '0');
	signal matrix_state_out : std_logic_vector(NMATRIX*8-1 downto 0) := (others => '0');
	signal digi_state_out	: std_logic_vector(15 downto 0)	 			:= (others => '0');
	
begin

reset_combined_n <= reset_n and pll_locked;
state_out <= matrix_state_out(7 downto 0) & digi_state_out;

mux_proc: process(reset_n, clk125) 
begin											-- for link enabling/disabling
	
	if(reset_n = '0') then 
		
		dataout	<= (others => '0');
		kout		<= (others => '0');
		syncout	<= (others => '0');
		
	elsif(rising_edge(clk125)) then
		
		for i in 0 to 3 loop -- link enabling/disabling
		
			syncout <= (others => '1');
			
			if(linken(i) = '1') then 
				dataout(i*8+7 downto i*8) 	<= data_fromfifo(i*8+7 downto i*8);
				kout(i)							<= data_fromfifo(32+i);
			else --linken(i)=0																-- send BC if link is disabled
				dataout(i*8+7 downto i*8) 	<= x"BC";
				kout(i)							<= '1';
			end if; --linken = '1'
		end loop;
		
	end if;

end process mux_proc; -- link enabling/disabling

	
my_fifo_wrap : fifo_data_gen_wrapper
	port map
	(	
		reset_n => reset_combined_n,
		
		clkin => clk125_loopback, 	-- wrclk
		datain => data_tofifo,
		
		clkout => clk125, 			-- rdclk
		dataout => data_fromfifo
		
	); -- my_fifo_wrap

my_MuPixDigital : MuPixDigitalTop	-- based on original Verilog Code
		port map (
			-- from matrix - Part A
			PriFromDet_PartA 		=> pri_fromdet(0),
			RowAddFromDet_PartA 	=> rowadd_fromdet(0*8+7 downto 0*8),
			ColAddFromDet_PartA 	=> coladd_fromdet(0*8+7 downto 0*8),
			TSFromDet_PartA 		=> ts_fromdet(0*16+15 downto 0*16),
			-- from matrix - Part B
			PriFromDet_PartB 		=> pri_fromdet(1),
			RowAddFromDet_PartB 	=> rowadd_fromdet(1*8+7 downto 1*8),
			ColAddFromDet_PartB 	=> coladd_fromdet(1*8+7 downto 1*8),
			TSFromDet_PartB 		=> ts_fromdet(1*16+15 downto 1*16),
			-- from matrix - Part C
			PriFromDet_PartC 		=> pri_fromdet(2),
			RowAddFromDet_PartC 	=> rowadd_fromdet(2*8+7 downto 2*8),
			ColAddFromDet_PartC 	=> coladd_fromdet(2*8+7 downto 2*8),
			TSFromDet_PartC 		=> ts_fromdet(2*16+15 downto 2*16),
			--to matrix
			TSToDet 			=> ts_todet,
			TSToDet2 		=> ts_todet2,
			-- to matrix - part A
			LdCol_PartA 	=> ldcol(0),
			RdCol_PartA 	=> rdcol(0),
			LdPix_PartA 	=> ldpix(0),
			PullDN_PartA 	=> pulldn(0),
			-- to matrix - part B
			LdCol_PartB 	=> ldcol(1),
			RdCol_PartB 	=> rdcol(1),
			LdPix_PartB 	=> ldpix(1),
			PullDN_PartB 	=> pulldn(1),
			-- to matrix - part C
			LdCol_PartC 	=> ldcol(2),
			RdCol_PartC 	=> rdcol(2),
			LdPix_PartC 	=> ldpix(2),
			PullDN_PartC 	=> pulldn(2),

			-- SM settings
			ckdivend 		=> ckdivend,			-- see Table 3 in "MuPix8DataFormat.pdf"
			ckdivend2 		=> ckdivend2,			-- see Table 3 in "MuPix8DataFormat.pdf"
			tsphase 			=> tsphase,				-- to adjust TS phase -- keep < ckdivend2
			timerend 		=> timerend,			-- SM runs at slower speed when timerend > 1 (keep active state for $timerend+1 cycle) -- see Table 3 in "MuPix8DataFormat.pdf"
			slowdownend 	=> slowdownend,		-- wait for hits or $slowdownend cycles before jumping to next state
			maxcycend 		=> maxcycend,			-- max number of hits that can be read out in one cycle
			resetckdivend 	=> resetckdivend,		-- length of sync sequence: N_res = 2*256*resetckdivend
			sendcounter 	=> sendcounter,		-- '1': debug mode --> send only counters
			
			linksel			=> linksel,				-- D: copy A,B,C or mux A,B,C

			Ser_res_n		=> reset_combined_n,
			RO_res_n 		=> reset_combined_n,
			syncRes 			=> syncres,

			-- data out
			d_out_A 		=> data_tofifo(7 downto 0),
			d_out_B 		=> data_tofifo(15 downto 8),
			d_out_C 		=> data_tofifo(23 downto 16),
			d_out_D 		=> data_tofifo(31 downto 24),
			k_out_A		=> data_tofifo(32),				-- concatenate data/k --> need only 1 fifo
			k_out_B		=> data_tofifo(33),
			k_out_C		=> data_tofifo(34),
			k_out_D		=> data_tofifo(35),
			
			reset_pll	=> reset_pll,
			pll_locked	=> pll_locked,
			digi_state_out => digi_state_out,

			-- clk in
			--clk_800p  	=> clk625,	 		-- for serializer and ClockGen (625 MHz)
			clk125_topll	=> clk125,
			-- clk out
			clk_8n 		=> clk_tomatrix, 		-- 62.5 MHz
			
			clkOut_4n	=> clk125_loopback,
			clkIn_4n 	=> clk125_loopback	-- 125 MHz
	
	); -- MuPixDigitalTop
	
	gen_matrix:							-- generate matrices to which digital part can talk (part A,B,C)
	for i in NMATRIX-1 downto 0 generate
		matrix_i : matrix_mp8
		generic map (
			NumCOL 		=> NumCOL(i),
			NumROW 		=> NumROW,
			MatrixSEL 	=> MatrixSEL(i)
		)
		port map (
			-- in
			rst_n 	=> reset_combined_n,
			syncres	=> syncres,
			clk 		=> clk_tomatrix,	-- 62.5 MHz
			mode		=> mode,
			slowdown	=> slowdown,
			numhits	=> numhits,
			
			ts_todet 	=> ts_todet,
			ts_todet2 	=> ts_todet2,
			ldcol 		=> ldcol(i),
			rdcol 		=> rdcol(i),
			ldpix 		=> ldpix(i),
			pulldn 		=> pulldn(i),
			
			timerend		=> timerend,
			-- out
			pri_fromdet		=> pri_fromdet(i),
			rowadd_fromdet => rowadd_fromdet(i*8+7 downto i*8),
			coladd_fromdet => coladd_fromdet(i*8+7 downto i*8),
			ts_fromdet		=> ts_fromdet(i*16+15 downto i*16),
			
			matrix_state_out 	=> matrix_state_out(i*8+7 downto i*8)
		);
		
	end generate; -- gen_matrix
	
end RTL;