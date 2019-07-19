-- Mupix 8 data path
-- Sebastian Dittmeier, 08.4.2017
-- dittmeier@physi.uni-heidelberg.de
-- Changed by Marius Koeppel 18.07.2019
-- makoeppe@students.uni-mainz.de


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_types.all;
use work.mupix_constants.all;
use work.mupix_registers.all;
use work.hitsorter_components.all;

entity data_path is
generic(
	NCHIPS: 				integer :=  4;
	NGX:	 				integer := 14;
	NLVDS: 				integer := 16;
	NSORTERINPUTS: 	integer :=  4	--up to 4 LVDS links merge to one sorter
);
port (
	resets_n:			in std_logic;--reg32;
	--resets: 			in reg32;
	--slowclk:			in std_logic;
	clk125:				in std_logic;
	counter125:			in reg64;
	
	--serial_data_in:	in std_logic_vector(NGX-1 downto 0);
	lvds_data_in:		in std_logic_vector(NLVDS-1 downto 0);
	
	--clkext_out:		out std_logic_vector(NCHIPS-1 downto 0);
	
	writeregs:			in reg32array;
	--regwritten:			in std_logic_vector(NREGISTERS-1 downto 0);

	readregs_slow: 		out reg32array;
	
	readmem_clk:		out std_logic;
	readmem_data:		out reg32;
	readmem_addr:		out std_logic_vector(15 downto 0);--readmemaddrtype;
	readmem_wren:		out std_logic;
	readmem_eoe:		out std_logic;
		
	-- trigger interface
	readtrigfifo:		out std_logic;
	fromtrigfifo:		in reg64;
	trigfifoempty:		in std_logic;
	
	readhitbusfifo:		out std_logic;
	fromhitbusfifo:		in reg64;
	hitbusfifoempty:	in std_logic--;	
);
end data_path;

architecture rtl of data_path is

-- synthesis booleans for different components
-- true  = entity is included in project 		(synthesis translate_on)
-- false = entity is not compiled in project (synthesis translate_off)
constant use_Hitsorter	: boolean := true;	-- false
constant use_MultiRO  	: boolean := true;	-- true
constant use_Histogram	: boolean := true;	-- true
constant use_LVDS  		: boolean := true;	-- false
constant use_SingleRO 	: boolean := true;	-- true
constant use_pseudodata	: boolean := true;	-- false

-- signals from Transceiver block (NGX)
signal rx_clkout_async		: std_logic_vector(NLVDS-1 downto 0);
signal rx_k_async			: std_logic_vector(NLVDS-1 downto 0);
signal rx_data_async		: std_logic_vector(NLVDS*8-1 downto 0);

signal rx_freqlocked_out	: std_logic_vector(NLVDS-1 downto 0);
signal rx_sync_out			: std_logic_vector(NLVDS-1 downto 0);
signal rx_err_out			: std_logic_vector(NLVDS-1 downto 0);
signal rx_disperr_out		: std_logic_vector(NLVDS-1 downto 0);
signal rx_pll_locked_out	: std_logic_vector(NLVDS-1 downto 0);
signal rx_patterndetect_out	: std_logic_vector(NLVDS-1 downto 0);
	
signal rx_runcounter		: links_reg32;
signal rx_errcounter		: links_reg32;

signal mux_gx_register		: std_logic_vector(1 downto 0);
signal mux_gx_lvds			: std_logic;
signal mux_lvds_clk			: std_logic_vector(NLVDS-1 downto 0);

-- signals from LVDS block (NLVDS)
signal lvds_clkout_async	: std_logic_vector(1 downto 0);
signal lvds_clkout_tofifo	: std_logic_vector(NLVDS-1 downto 0);
signal lvds_k_async			: std_logic_vector(NLVDS-1 downto 0);
signal lvds_data_async		: std_logic_vector(NLVDS*8-1 downto 0);

signal lvds_syncstatus		: std_logic_vector(NLVDS-1 downto 0);
signal lvds_pll_locked		: std_logic_vector(1 downto 0);
signal lvds_dpa_locked		: std_logic_vector(NLVDS-1 downto 0);
	
signal lvds_runcounter		: links_reg32;
signal lvds_errcounter		: links_reg32;	

-- signals from syncfifo to mux
signal rx_k_sync			: std_logic_vector(NLVDS-1 downto 0);
signal rx_data_sync			: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_data_valid_sync	: std_logic_vector(NLVDS-1 downto 0);
signal lvds_k_sync			: std_logic_vector(NLVDS-1 downto 0);
signal lvds_data_sync		: std_logic_vector(NLVDS*8-1 downto 0);
signal lvds_data_valid_sync	: std_logic_vector(NLVDS-1 downto 0);

-- signals after mux
signal rx_data				: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_k					: std_logic_vector(NLVDS-1 downto 0);
signal data_valid			: std_logic_vector(NLVDS-1 downto 0);

-- signals to the unpacker
signal rx_k_in				: std_logic_vector(NLVDS-1 downto 0);
signal rx_data_in			: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_syncstatus_in		: std_logic_vector(NLVDS-1 downto 0);
signal rx_syncstatus_nomask	: std_logic_vector(NLVDS-1 downto 0);

-- signals from the MP8 data generator, to muxed with data inputs
signal rx_data_gen			: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_k_gen				: std_logic_vector(NLVDS-1 downto 0);
signal rx_syncstatus_gen	: std_logic_vector(NLVDS-1 downto 0);

signal gen_state_out		: std_logic_vector(NCHIPS*24-1 downto 0);
signal gen_linken			: std_logic_vector(NCHIPS*4-1 downto 0);
-- signal to mux between actual data and on-FPGA generated data
signal mux_fake_data		: std_logic := '0';

-- hits + flag to indicate a word as a hit, after unpacker
signal hits 				: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal hits_ena				: std_logic_vector(NLVDS-1 downto 0);
signal hits_i 				: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal hits_ena_i			: std_logic_vector(NLVDS-1 downto 0);

signal hits_i_mp8 				: std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
signal hits_ena_i_mp8 			: std_logic;
signal coarsecounters_i_mp8 	: std_logic_vector(COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_i_mp8	: std_logic;
signal unpack_errorcounter_mp8 	: reg32;
signal link_flag_i_mp8			: std_logic;

-- hits after gray-decoding
signal binhits_TS		: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena_TS	: std_logic_vector(NLVDS-1 downto 0);
signal binhits 			: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena		: std_logic_vector(NLVDS-1 downto 0);
signal binhits_mp8 		: std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena_mp8	: std_logic;

-- hits to sorter
signal sorterhits 		: std_logic_vector(4*NSORTERINPUTS*UNPACKER_HITSIZE-1 downto 0);
signal sorterhits_ena	: std_logic_vector(4*NSORTERINPUTS-1 downto 0);
signal muxbinhit		: std_logic_vector(MHITSIZE-1 downto 0);
signal muxbinhitena		: std_logic;
signal muxcounterena	: std_logic;

-- flag to indicate link, after unpacker
signal link_flag 		: std_logic_vector(NLVDS-1 downto 0);
signal link_flag_i		: std_logic_vector(NLVDS-1 downto 0);

-- link flag is pipelined once because hits are gray decoded
signal link_flag_del 	: std_logic_vector(NLVDS-1 downto 0);

-- link flag is pipelined twice because hits are gray decoded
signal link_flag_reg 	: std_logic_vector(NLVDS-1 downto 0);

-- counter + flag to indicate word as a counter, after unpacker
signal coarsecounters 		: std_logic_vector(NLVDS*COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena 	: std_logic_vector(NLVDS-1 downto 0);
signal coarsecounters_i   	: std_logic_vector(NLVDS*COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_i	: std_logic_vector(NLVDS-1 downto 0);

-- counter is pipelined once because hits are gray decoded
signal coarsecounters_del 		: std_logic_vector(NLVDS *COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_del 	: std_logic_vector(NLVDS-1 downto 0);

-- counter is pipelined twice because hits are gray decoded
signal coarsecounters_reg 		: std_logic_vector(NLVDS *COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_reg 	: std_logic_vector(NLVDS-1 downto 0);

-- counter for sorter
signal sortercoarsecounters 	: std_logic_vector(4*NSORTERINPUTS*COARSECOUNTERSIZE-1 downto 0);
signal sortercoarsecounters_ena	: std_logic_vector(4*NSORTERINPUTS-1 downto 0);

-- muxed hits for single chip RO
signal single_hits 					: std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
signal single_hits_ena 				: std_logic;
signal single_coarsecounters 		: std_logic_vector(COARSECOUNTERSIZE-1 downto 0);
signal single_coarsecounters_ena	: std_logic;
signal single_link_flag 			: std_logic;	
signal single_chipmarker			: std_logic_vector(7 downto 0);

-- this is the mux
signal single_res			: std_logic_vector(COARSECOUNTERSIZE+UNPACKER_HITSIZE+3-1 downto 0);
type single_mux_array is array (NLVDS-1 downto 0) of std_logic_vector(COARSECOUNTERSIZE+UNPACKER_HITSIZE+3-1 downto 0);
signal single_muxdata 		: single_mux_array;

-- signals between hit serializer and sortersignal 
signal hit_ser_out		: hit_array_t;
signal hit_ser_ena		: STD_LOGIC_VECTOR(NSORTERINPUTS-1 downto 0);
signal time_ser_out		: counter_array_t;
signal time_ser_ena		: STD_LOGIC_VECTOR(NSORTERINPUTS-1 downto 0);
signal hits_to_sorter	: hit_array_t;

-- signals for the hit counter (pre and post sorter)
signal sorter_out_hit	: std_logic;

-- RO mode ouputs
-- single chip RO
signal single2mem		: reg32;
signal single2memena	: std_logic;
signal single2memeoe	: std_logic;
-- single chip RO with zero suppression
signal singlezs2mem		: reg32;
signal singlezs2memena	: std_logic;
signal singlezs2memeoe	: std_logic;
-- telescope RO
signal telescope2mem	: reg32;
signal telescope2memena	: std_logic;
signal telescope2memeoe	: std_logic;
-- multichip RO with zero suppression
signal allsinglezs2mem		: reg32;
signal allsinglezs2memena	: std_logic;
signal allsinglezs2memeoe	: std_logic;

-- RO mode selection and muxing
signal romode			: std_logic_vector(2 downto 0);

signal muxdata0			: reg32;
signal muxdata1			: reg32;
signal muxdata2			: reg32;
signal muxdata3			: reg32;
signal muxdata4			: reg32;
signal muxdata5			: reg32;
signal muxdata6			: reg32;
signal muxdata7			: reg32;

signal muxena0			: std_logic;
signal muxena1			: std_logic;
signal muxena2			: std_logic;
signal muxena3			: std_logic;
signal muxena4			: std_logic;
signal muxena5			: std_logic;
signal muxena6			: std_logic;
signal muxena7			: std_logic;

signal muxeoe0			: std_logic;
signal muxeoe1			: std_logic;
signal muxeoe2			: std_logic;
signal muxeoe3			: std_logic;
signal muxeoe4			: std_logic;
signal muxeoe5			: std_logic;
signal muxeoe6			: std_logic;
signal muxeoe7			: std_logic;

-- signals to the PCIE memory
signal muxtomem				: reg32;
signal muxenatomem			: std_logic;
signal muxeoetomem			: std_logic;
signal readmem_addr_reg 	: std_logic_vector(15 downto 0);--readmemaddrtype;
signal readmem_eoeaddr_reg	: std_logic_vector(15 downto 0);--readmemaddrtype;


-- histograms!
signal histo_komma_out				: std_logic_vector(47 downto 0);
signal histo_counter_out			: reg64array;
signal histo_hits_row				: chips_reg32;
signal histo_hits_col				: chips_reg32;
signal histo_hits_ts				: chips_reg32;
signal histo_hits_matrix			: chips_reg32;
signal histo_hits_multi				: chips_reg32;
signal histo_timediff				: chips_reg32;
signal histo_hitdiff				: chips_reg32;
signal histo_hitTSdiff				: chips_reg32;
signal histo_hitTSdiff_4mux1		: reg32;
signal histo_timediff_col			: chips_histo_reg32;
signal histo_timediff_row			: chips_histo_reg32;

-- hit counter pre- and postsort
signal presort_sum		: 		std_logic_vector(47 downto 0);
signal postsort_sum	: 		std_logic_vector(47 downto 0);
type   serializer_reg48_array is array (NSORTERINPUTS-1 downto 0) of std_logic_vector(47 downto 0);
signal presort_per_serializer	: serializer_reg48_array;
signal presort_sum_r_0 : std_logic_vector(47 downto 0);
signal presort_sum_r_1 : std_logic_vector(47 downto 0);

--signal sync_fifo_mux_sel: std_logic_vector(2 downto 0);
signal sort_mux_sel		: std_logic_vector(2 downto 0);

-- error signal output from unpacker
signal unpack_errorout		: links_reg32;
signal unpack_errorcounter	: links_reg32;
signal unpack_readycounter	: links_reg32;

-- muxing trigger fifo for RO modes
signal muxreadtrigfifo0		: std_logic;
signal muxreadtrigfifo1		: std_logic;
signal muxreadtrigfifo2		: std_logic;
signal muxreadtrigfifo3		: std_logic;
signal muxreadtrigfifo4		: std_logic;
signal muxreadtrigfifo5		: std_logic;
signal muxreadtrigfifo6		: std_logic;
signal muxreadtrigfifo7		: std_logic;

-- trigger fifo signals for RO modes
signal telescope_readtrigfifo		: std_logic;
signal telescope_trigfifoempty		: std_logic;
signal allsinglezs2_readtrigfifo		: std_logic;
signal allsinglezs2_trigfifoempty	: std_logic;

-- hitbus interface
signal muxreadhitbusfifo0			: std_logic;
signal muxreadhitbusfifo1			: std_logic;
signal muxreadhitbusfifo2			: std_logic;
signal muxreadhitbusfifo3			: std_logic;
signal muxreadhitbusfifo4			: std_logic;
signal muxreadhitbusfifo5			: std_logic;
signal muxreadhitbusfifo6			: std_logic;
signal muxreadhitbusfifo7			: std_logic;

-- hitbus fifo signals for RO modes
signal telescope_readhitbusfifo		: std_logic;
signal telescope_hitbusfifoempty		: std_logic;
signal allsinglezs2_readhitbusfifo	: std_logic;
signal allsinglezs2_hitbusfifoempty	: std_logic;

-- writeregisters are registered once to reduce long combinational paths
signal writeregs_reg				: reg32array;
--signal regwritten_reg				: std_logic_vector(NREGISTERS-1 downto 0); 

-- hitsorter resets and debug signals
signal cnt_hitswritten		:  reg64;
signal cnt_hitsread			:  reg64;
signal cnt_overflow			:  reg64;
signal cnt_coarse				:  reg64;
signal readregs_muxed_gx 	: reg32array;
signal readregs_muxed_lvds : reg32array;

signal is_atlaspix 			: std_logic_vector(NLVDS-1 downto 0);
signal is_mp7		 			: std_logic;

begin

writregs_clocking : process(clk125)
begin
	if(rising_edge(clk125))then
		for I in 63 downto 0 loop
			writeregs_reg(I)	<= writeregs(I);
			--regwritten_reg(I)	<= regwritten(I);		
		end loop;
	end if;
end process writregs_clocking;

------------------------------------------------------------------------------------
---------------------- LVDS Receiver part ------------------------------------------
lvds_block	: work.receiver_block_mupix
	generic map(
		NINPUT	=> NLVDS,
		NCHIPS	=> NCHIPS
	)
	port map(
		reset_n				=> resets_n,
		reset_n_errcnt		=> resets_n,
		rx_in					=> lvds_data_in,
		rx_inclock			=> clk125,
		rx_data_bitorder	=> writeregs(RECEIVER_REGISTER_W)(LVDS_ORDER_BIT),	-- set to '0' for data as received, set to '1' to invert order LSB to MSB
		rx_state				=> open,--readregs_slow(LVDS_REGISTER2_R),
		rx_ready				=> lvds_syncstatus,
		rx_data				=> lvds_data_async,
		rx_k					=> lvds_k_async,
		rx_clkout			=> lvds_clkout_async,
		pll_locked			=> lvds_pll_locked,	-- write to some register!
		rx_dpa_locked_out	=> lvds_dpa_locked,
	
		rx_runcounter		=> lvds_runcounter,
		rx_errorcounter	=> lvds_errcounter	-- would be nice to add some error counter
);
			
gen_lvds_reg:
for i in NCHIPS-1 downto 0 generate
	with mux_gx_register select readregs_muxed_lvds(RECEIVER_RUNTIME_REGISTER_R+i) <=
		lvds_runcounter(i*4) 	when "00",
		lvds_runcounter(i*4+1) 	when "01",
		lvds_runcounter(i*4+2)	when "10",
		lvds_runcounter(i*4+3) 	when "11";	
		
	with mux_gx_register select readregs_muxed_lvds(RECEIVER_ERRCOUNT_REGISTER_R+i) <=
		lvds_errcounter(i*4) 	when "00",
		lvds_errcounter(i*4+1) 	when "01",
		lvds_errcounter(i*4+2)	when "10",
		lvds_errcounter(i*4+3) 	when "11";			
end generate gen_lvds_reg;

--readregs_slow(LVDS_REGISTER_R)(DPALOCKED_RANGE) 		<= lvds_dpa_locked; 
--readregs_slow(LVDS_REGISTER_R)(RXREADY_RANGE) 			<= lvds_syncstatus;
--readregs_slow(LVDS_REGISTER2_R)(LVDS_PLL_LOCKED_RANGE)	<= lvds_pll_locked;

lvds_clkout_tofifo(15 downto 8) 	<= (others => lvds_clkout_async(1));
lvds_clkout_tofifo(7 downto 0) 	<= (others => lvds_clkout_async(0));


--ToDo sync data to 125 MHz clk
--gen_sync_fifo:
--for i in NCHIPS-1 downto 0 generate
--	sf_lvds: work.syncfifo
--		port map(
--			aclr		=> not resets,
--			data		=> lvds_data_async(NLVDS*8-1),: IN STD_LOGIC_VECTOR (8 DOWNTO 0);
--			rdclk		=> clk125,
--			rdreq		: IN STD_LOGIC ;
--			wrclk		: IN STD_LOGIC ;
--			wrreq		=> lvds_syncstatus(i),
--			q		: OUT STD_LOGIC_VECTOR (8 DOWNTO 0);
--			rdusedw		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
--			wrusedw		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0)
--		
--		
--			ready				=> lvds_syncstatus,
--			clkout			=> clk125,
--			reset_n			=> resets_n,
--			
--			clkin				=> lvds_clkout_tofifo,
--			datain			=> lvds_data_async,
--			kin				=> lvds_k_async,
--			
--			dataout			=> lvds_data_sync,
--			kout				=> lvds_k_sync,
--			data_valid		=> lvds_data_valid_sync
--	);
--end generate;

rx_data 	 	<= lvds_data_sync;
rx_k_sync 	<= lvds_k_sync;
data_valid 	<= lvds_data_valid_sync;

-- clock this can be used for datagenerator
mux_process_datagenerator : process(clk125)
begin
	if(rising_edge(clk125))then
--		mux_fake_data <= writeregs_reg(DATA_GEN_REGISTER_W)(MUX_DATA_GEN_BIT);
--		if(mux_fake_data = '0')then
			rx_data_in				<= rx_data;
			rx_k_in					<= rx_k;
			rx_syncstatus_nomask	<= data_valid;		
--		else
--			rx_data_in				<= rx_data_gen;	
--			rx_k_in					<= rx_k_gen;	
--			rx_syncstatus_nomask	<= rx_syncstatus_gen;				
--		end if;
	end if;
end process mux_process_datagenerator;

rx_syncstatus_in <= rx_syncstatus_nomask;



------------------------------------------------------------------------------------
------------------- Unpack the data ------------------------------------------------
gen_unpack_reg:
for i in NCHIPS-1 downto 0 generate
with mux_gx_register select readregs_slow(RECEIVER_UNPACKERRCOUNT_REGISTER_R+i) <=
		unpack_errorcounter(i*4) 	when "00",
		unpack_errorcounter(i*4+1) 	when "01",
		unpack_errorcounter(i*4+2)	when "10",
		unpack_errorcounter(i*4+3) 	when "11";	
end generate gen_unpack_reg;	

genunpack:
for i in NLVDS-1 downto 0 generate	-- if logic utilization becomes a problem, build triple-unpacker only for the shared links
-- experimental: move to unpacker triple only for links 3, 7, 11 and 15
	unpacker_trips : work.data_unpacker_triple_new
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n,
			clk					=> clk125,
			datain				=> rx_data_in(i*8+7 downto i*8),
			kin					=> rx_k_in(i),
			readyin				=> rx_syncstatus_in(i),
			is_shared			=> writeregs_reg(LINK_REGISTER_W)(i),
			is_atlaspix			=> is_atlaspix(i),
			hit_out				=> hits_i(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena				=> hits_ena_i(i),
			coarsecounter		=> coarsecounters_i(COARSECOUNTERSIZE*(i+1)-1 downto COARSECOUNTERSIZE*i),
			coarsecounter_ena	=> coarsecounters_ena_i(i),
			link_flag			=> link_flag_i(i),
			errorcounter		=> unpack_errorcounter(i)--,
	);
		
	degray_trip : work.hit_ts_conversion 
		port map(
			reset_n				=> resets_n,
			clk					=> clk125, 
			invert_TS			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_INVERT_BIT),
			invert_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT),
			gray_TS				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			gray_TS2				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_in			=> hits_ena(i),
			hit_out				=> binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_out			=> binhits_ena(i)--,
		);
end generate genunpack;

-- register output from data unpacker because of timing issues
process (clk125, resets_n)
begin
	if resets_n='0' then
		hits					<= (others => '0');
		hits_ena				<= (others => '0');
		coarsecounters			<= (others => '0');
		coarsecounters_ena		<= (others => '0');
		link_flag				<= (others => '0');
	elsif rising_edge(clk125) then
		hits					<= hits_i;
		hits_ena				<= hits_ena_i;
		coarsecounters			<= coarsecounters_i;
		coarsecounters_ena		<= coarsecounters_ena_i;
		link_flag				<= link_flag_i;	
	end if;
end process;

-- delay cc by one cycle to be in line with hit
-- Seb: new degray - back to one
process(clk125)
begin
if(clk125'event and clk125 = '1') then
	coarsecounters_del		<= coarsecounters;
	coarsecounters_ena_del	<= coarsecounters_ena;
	link_flag_del				<= link_flag;
end if;
end process;	

sort_mux_sel <= writeregs_reg(DEBUG_CHIP_SELECT_REGISTER_W)(CHIP_SELECT_RANGE);	



---------------------------------------------------------------------------------------------------		
------------------------ For all chips: Single chip RO mode, zero suppressed ----------------------
-- now we also try to include the muxed links!
multirozs : work.multichip_ro_zerosupressed
	generic map(
		COARSECOUNTERSIZE		=> COARSECOUNTERSIZE,
		HITSIZE					=> UNPACKER_HITSIZE,
		NCHIPS 					=> NLVDS	 
	)
	port map(
		clk						=> clk125,												
		reset_n					=> resets_n,		
		counter125				=> counter125,
		link_flag				=> link_flag_del,		
		hits_in					=> binhits,			
		hits_ena					=> binhits_ena,	
		coarsecounters			=> coarsecounters_del,
		coarsecounters_ena	=> coarsecounters_ena_del,
		chip_markers			=> writeregs_reg(RO_CHIPMARKER_REGISTER_W),
		prescale					=> writeregs_reg(RO_PRESCALER_REGISTER_W)(RO_PRESCALER_RANGE),
		is_shared				=> writeregs_reg(LINK_REGISTER_W)(LINK_SHARED_RANGE),		
		tomemdata				=> allsinglezs2mem,
		tomemena					=> allsinglezs2memena,
		tomemeoe					=> allsinglezs2memeoe,
		errcounter_overflow 	=> readregs_slow(MULTICHIP_RO_OVERFLOW_REGISTER_R),
		errcounter_sel_in		=> writeregs_reg(DEBUG_CHIP_SELECT_REGISTER_W)(CHIPRANGE-1 downto 0),
		readtrigfifo			=> allsinglezs2_readtrigfifo,
		fromtrigfifo			=> fromtrigfifo,
		trigfifoempty			=> allsinglezs2_trigfifoempty,
		readhbfifo				=> allsinglezs2_readhitbusfifo,
		fromhbfifo				=> fromhitbusfifo,
		hbfifoempty				=> allsinglezs2_hitbusfifoempty--,
);



---------------------------------------------------------------------------------------------------		
---------------------------------------------- Hitsorter ------------------------------------------
--reduced sorter
gen_sorterhits:	-- NSORTERINPUTS = 1, does not work properly otherwise! -- only links A
for i in 0 to NSORTERINPUTS-1 generate
	sorterhits((4*i+1)*UNPACKER_HITSIZE-1 downto (4*i)*UNPACKER_HITSIZE)				<= binhits((16*i+1)*UNPACKER_HITSIZE-1 downto (16*i)*UNPACKER_HITSIZE);			-- link A of 1st and 5th chip
	sorterhits((4*i+2)*UNPACKER_HITSIZE-1 downto (4*i+1)*UNPACKER_HITSIZE)				<= binhits((16*i+5)*UNPACKER_HITSIZE-1 downto (16*i+4)*UNPACKER_HITSIZE);		-- link A of 2nd and 6th chip
	sorterhits((4*i+3)*UNPACKER_HITSIZE-1 downto (4*i+2)*UNPACKER_HITSIZE)				<= binhits((16*i+9)*UNPACKER_HITSIZE-1 downto (16*i+8)*UNPACKER_HITSIZE);		-- link A of 3rd and 7th chip
	sorterhits((4*i+4)*UNPACKER_HITSIZE-1 downto (4*i+3)*UNPACKER_HITSIZE)				<= binhits((16*i+13)*UNPACKER_HITSIZE-1 downto (16*i+12)*UNPACKER_HITSIZE);	-- link A of 4th and 8th chip
	sorterhits_ena(4*i)																	<= binhits_ena(16*i);	
	sorterhits_ena(4*i+1)																<= binhits_ena(16*i+4);	
	sorterhits_ena(4*i+2)																<= binhits_ena(16*i+8);
	sorterhits_ena(4*i+3)																<= binhits_ena(16*i+12);
	sortercoarsecounters((4*i+1)*COARSECOUNTERSIZE-1 downto (4*i)*COARSECOUNTERSIZE)	<= coarsecounters_del((16*i+1)*COARSECOUNTERSIZE-1 downto (16*i)*COARSECOUNTERSIZE);
	sortercoarsecounters((4*i+2)*COARSECOUNTERSIZE-1 downto (4*i+1)*COARSECOUNTERSIZE)	<= coarsecounters_del((16*i+5)*COARSECOUNTERSIZE-1 downto (16*i+4)*COARSECOUNTERSIZE);
	sortercoarsecounters((4*i+3)*COARSECOUNTERSIZE-1 downto (4*i+2)*COARSECOUNTERSIZE)	<= coarsecounters_del((16*i+9)*COARSECOUNTERSIZE-1 downto (16*i+8)*COARSECOUNTERSIZE);
	sortercoarsecounters((4*i+4)*COARSECOUNTERSIZE-1 downto (4*i+3)*COARSECOUNTERSIZE)	<= coarsecounters_del((16*i+13)*COARSECOUNTERSIZE-1 downto (16*i+12)*COARSECOUNTERSIZE);	
	sortercoarsecounters_ena(4*i)														<= coarsecounters_ena_del(16*i);	
	sortercoarsecounters_ena(4*i+1)														<= coarsecounters_ena_del(16*i+4);	
	sortercoarsecounters_ena(4*i+2)														<= coarsecounters_ena_del(16*i+8);
	sortercoarsecounters_ena(4*i+3)														<= coarsecounters_ena_del(16*i+12);
end generate;

gen_hitser:	-- the 4 MSB of hit_in and time_in encode the link number as chip marker
for i in 0 to NSORTERINPUTS-1 generate
	hit_ser : work.hit_serializer
	generic map(
		SERIALIZER_HITSIZE 	=> UNPACKER_HITSIZE+4,	-- 4 additional bits to encode the receiver
		SERBINCOUNTERSIZE	=> BINCOUNTERSIZE+4,
		SERHITSIZE			=> 40
	)
	port map(
		reset_n				=> resets_n,
		clk					=> clk125,
		hit_in1				=> std_logic_vector(to_unsigned(4*i  ,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+1)-1 downto UNPACKER_HITSIZE*4*i),
		hit_ena1			=> sorterhits_ena(4*i),
		hit_in2				=> std_logic_vector(to_unsigned(4*i+1,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+2)-1 downto UNPACKER_HITSIZE*(4*i+1)),
		hit_ena2			=> sorterhits_ena(4*i+1),
		hit_in3				=> std_logic_vector(to_unsigned(4*i+2,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+3)-1 downto UNPACKER_HITSIZE*(4*i+2)),
		hit_ena3			=> sorterhits_ena(4*i+2),
		hit_in4				=> std_logic_vector(to_unsigned(4*i+3,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+4)-1 downto UNPACKER_HITSIZE*(4*i+3)),
		hit_ena4			=> sorterhits_ena(4*i+3),
		time_in1			=> std_logic_vector(to_unsigned(4*i  ,4)) & sortercoarsecounters(COARSECOUNTERSIZE*4*i+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*4*i),
		time_ena1			=> sortercoarsecounters_ena(4*i),
		time_in2			=> std_logic_vector(to_unsigned(4*i+1,4)) & sortercoarsecounters(COARSECOUNTERSIZE*(4*i+1)+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*(4*i+1)),
		time_ena2			=> sortercoarsecounters_ena(4*i+1),
		time_in3			=> std_logic_vector(to_unsigned(4*i+2,4)) & sortercoarsecounters(COARSECOUNTERSIZE*(4*i+2)+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*(4*i+2)),
		time_ena3			=> sortercoarsecounters_ena(4*i+2),
		time_in4			=> std_logic_vector(to_unsigned(4*i+3,4)) & sortercoarsecounters(COARSECOUNTERSIZE*(4*i+3)+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*(4*i+3)),
		time_ena4			=> sortercoarsecounters_ena(4*i+3),
		hit_out				=> hit_ser_out(i),
		hit_ena				=> hit_ser_ena(i),
		time_out			=> time_ser_out(i),
		time_ena			=> time_ser_ena(i)--,
	);
end generate;

-- count number of hits before sorter
gen_pre_sort_cnt:
for i in 0 to NSORTERINPUTS-1 generate
	pre_sort_cnt : work.hit_counter_simple 
	port map( 
		clock 		=> clk125,	-- should be the same as readclk of hitsorter
		reset_n 	=> resets_n,
		coarse_ena	=> time_ser_ena(i),
		hits_ena_in	=> hit_ser_ena(i),
		counter		=> presort_per_serializer(i)
	);
end generate;

readregs_slow(RECEIVER_PRESORT_TOP_REGISTER_R)(15 downto 0)	<= presort_per_serializer(0)(47 downto 32);--presort_sum(47 downto 32);
readregs_slow(RECEIVER_PRESORT_BOTTOM_REGISTER_R)			<= presort_per_serializer(0)(REG64_BOTTOM_RANGE);--presort_sum(REG64_BOTTOM_RANGE);

sort : work.hitsorter 
	generic map(
		NINPUT			=> 1,
		NSORTERINPUTS 	=> 1--,
--		NINPUT2		=> 2,
--		NINPUT4		=> 1,
--		NINPUT8		=> 1,
--		NINPUT16	=> 1,
--		HITSIZE		=> 2+2+8+8+6+10	-- serializer + link + Row + Col + Charge + TS	
	)
	port map(
		reset_n			=> resets_n,
		writeclk		=> clk125,
		tsdivider		=> writeregs_reg(HITSORTER_TIMING_REGISTER_W)(HITSORTER_TSDIVIDER_RANGE),	--"0000",				-- TODO: to be set via register
		readclk			=> clk125,		
		hits_ser		=> hit_ser_out,
		hits_ser_en		=> hit_ser_ena,
		ts_en			=> time_ser_ena,
		tsdelay			=> writeregs_reg(HITSORTER_TIMING_REGISTER_W)(HITSORTER_TSDELAY_RANGE),	-- "1010000000",	-- "0000000000"	-- TODO: to be set via register
		fromtrigfifo	=> fromtrigfifo,
		trigfifoempty	=> telescope_trigfifoempty,
		fromhbfifo		=> fromhitbusfifo,
		hbfifoempty		=> telescope_hitbusfifoempty,
		data_out		=> telescope2mem,					
		out_ena			=> telescope2memena,
		out_eoe			=> telescope2memeoe,
		out_hit			=> sorter_out_hit,
		readtrigfifo	=> telescope_readtrigfifo,
		readhbfifo		=> telescope_readhitbusfifo,
		received_hits	=> readregs_slow(HITSORTER_RECEIVED_HITS_REGISTER_R),
		outoftime_hits	=> readregs_slow(HITSORTER_OUTOFTIME_HITS_REGISTER_R),
		intime_hits		=> readregs_slow(HITSORTER_INTIME_HITS_REGISTER_R),
		memwrite_hits	=> readregs_slow(HITSORTER_MEMWRITE_HITS_REGISTER_R),
		overflow_hits	=> readregs_slow(HITSORTER_OVERFLOW_HITS_REGISTER_R),
		sent_hits		=> readregs_slow(HITSORTER_SENT_HITS_REGISTER_R),
		break_counter	=> readregs_slow(HITSORTER_BREAKCOUNTER_REGISTER_R)--,
);
	
-- count number of hits after sorter
post_sort_cnt : work.hit_counter_simple 
	port map( 
		clock 		=> clk125,	-- should be the same as readclk of hitsorter
		reset_n 	=> resets_n,
		coarse_ena	=> '0',
		hits_ena_in	=> sorter_out_hit,
		counter		=> postsort_sum	-- TODO: write postsort_sum to a register
);
	
readregs_slow(RECEIVER_POSTSORT_TOP_REGISTER_R)(15 downto 0) <= postsort_sum(47 downto 32);
readregs_slow(RECEIVER_POSTSORT_BOTTOM_REGISTER_R)			 <= postsort_sum(REG64_BOTTOM_RANGE);



------------------------------------------------------------------------------------------------		
-- I guess the code below does not require any changes, unless new readout modes are implemented
------------------------------------------------------------------------------------------------

------------------------------------------TRIGGER LOGiC-----------------------------------------
telescope_trigfifoempty		<= trigfifoempty  when romode=3 else '1';
allsinglezs2_trigfifoempty	<= trigfifoempty  when romode=4 else '1';

readtrigfifo <= muxreadtrigfifo0	when romode=0 else
				muxreadtrigfifo1	when romode=1 else
				muxreadtrigfifo2	when romode=2 else
				muxreadtrigfifo3	when romode=3 else
				muxreadtrigfifo4	when romode=4 else
				muxreadtrigfifo5	when romode=5 else
				muxreadtrigfifo6	when romode=6 else
				muxreadtrigfifo7	when romode=7 else
				'0';
						
muxreadtrigfifo0	<= '0';
muxreadtrigfifo1	<= '0';
muxreadtrigfifo2	<= '0';
muxreadtrigfifo3 	<= telescope_readtrigfifo;
muxreadtrigfifo4	<= allsinglezs2_readtrigfifo;
muxreadtrigfifo5	<= '0';
muxreadtrigfifo6	<= '0';
muxreadtrigfifo7	<= '0';
------------------------------------------TRIGGER LOGiC-----------------------------------------


------------------------------------------HITBUS LOGiC------------------------------------------
-- only selected RO mode sees if there are triggers, the other modes just skip
telescope_hitbusfifoempty			<= hitbusfifoempty  when romode=3 else '1';
allsinglezs2_hitbusfifoempty		<= hitbusfifoempty  when romode=4 else '1';

-- only selected readout mode may read from the FIFO
readhitbusfifo	<=	muxreadhitbusfifo0	when romode=0 else
					muxreadhitbusfifo1	when romode=1 else
					muxreadhitbusfifo2	when romode=2 else
					muxreadhitbusfifo3	when romode=3 else
					muxreadhitbusfifo4	when romode=4 else
					muxreadhitbusfifo5	when romode=5 else
					muxreadhitbusfifo6	when romode=6 else
					muxreadhitbusfifo7	when romode=7 else
					'0';

muxreadhitbusfifo0	<= '0';
muxreadhitbusfifo1	<= '0';
muxreadhitbusfifo2	<= '0';						
muxreadhitbusfifo3 	<= telescope_readhitbusfifo;
muxreadhitbusfifo4	<= allsinglezs2_readhitbusfifo;
muxreadhitbusfifo5	<= '0';
muxreadhitbusfifo6	<= '0';
muxreadhitbusfifo7	<= '0';
------------------------------------------HITBUS LOGiC------------------------------------------


--------------------------------------Multiplex output and drive memory-------------------------
-- Readout modes
romode		<= writeregs_reg(RO_MODE_REGISTER_W)(RO_MODE_RANGE);
-- 0: Raw data from links (k signal is lost) / 4 * 8 bits
with mux_gx_register select muxdata0 <=
		rx_data_in(31 downto 0) 	when "00",
		rx_data_in(63 downto 32) 	when "01",
		rx_data_in(95 downto 64)	when "10",
		rx_data_in(127 downto 96)	when "11";	
muxena0	<= '1';	-- when rx_syncstatus(0) = '1' or rx_syncstatus(1) = '1' or rx_syncstatus(2) = '1' or rx_syncstatus(3) = '1'
							-- else '0';
muxeoe0	<= '0';					

-- 1: Single chip (A front) readout
muxdata1	<= single2mem;
muxena1		<= single2memena;
muxeoe1		<= single2memeoe;

-- 2: Single chip (A front) readout, zero suppressed
muxdata2	<= singlezs2mem;
muxena2		<= singlezs2memena;
muxeoe2		<= singlezs2memeoe;

-- 3: Telescope readout
muxdata3	<= telescope2mem;
muxena3		<= telescope2memena;
muxeoe3		<= telescope2memeoe;


-- 4: Multiplexed Hitblocks, using single readout of all chips, zero suppressed
muxdata4	<= allsinglezs2mem;
muxena4		<= allsinglezs2memena;
muxeoe4		<= allsinglezs2memeoe;

-- 5: Not used
muxdata5	<= (others => '0');
muxena5		<= '0';
muxeoe5		<= '0';
-- 6: Not used
muxdata6	<= (others => '0');
muxena6		<= '0';
muxeoe6		<= '0';
-- 7: Not used
muxdata7	<= (others => '0');
muxena7		<= '0';
muxeoe7		<= '0';

memmux : work.memorymux 
	port map(
		clock	=> clk125,
		data0x	=> muxdata0,
		data1x	=> muxdata1,
		data2x	=> muxdata2,
		data3x	=> muxdata3,
		data4x	=> muxdata4,
		data5x	=> muxdata5,
		data6x	=> muxdata6,
		data7x	=> muxdata7,
		sel		=> romode,
		result	=> muxtomem--,
);

enamux : work.enablemux 
	port map(
		clock	=> clk125,
		data0	=> muxena0,
		data1	=> muxena1,
		data2	=> muxena2,
		data3	=> muxena3,
		data4	=> muxena4,
		data5	=> muxena5,
		data6	=> muxena6,
		data7	=> muxena7,
		sel		=> romode,
		result	=> muxenatomem--,
);

eoemux : work.enablemux 
	port map(
		clock   => clk125,
		data0	=> muxeoe0,
		data1	=> muxeoe1,
		data2	=> muxeoe2,
		data3	=> muxeoe3,
		data4	=> muxeoe4,
		data5	=> muxeoe5,
		data6	=> muxeoe6,
		data7	=> muxeoe7,
		sel		=> romode,
		result	=> muxeoetomem--,
);	
	
-- Write to memory
readmem_addr <= readmem_addr_reg;
readmem_clk	 <= clk125;

readregs_slow(MEM_ADDRESS_REGISTER_R)(WRITE_ADDRESS_RANGE)	<= readmem_addr_reg;
readregs_slow(MEM_ADDRESS_REGISTER_R)(EOE_ADDRESS_RANGE)	<= readmem_eoeaddr_reg;

process(resets_n, clk125)
begin
if(resets_n = '0') then
	readmem_addr_reg				<= (others => '0');
	readmem_wren					<= '0';
	readmem_eoe						<= '0';
	readmem_eoeaddr_reg				<= (others => '0');
elsif(clk125'event and clk125 = '1') then
	if(writeregs_reg(RO_MODE_REGISTER_W)(RO_ENABLE_BIT) = '1') then
		readmem_data				<= muxtomem;
		readmem_wren				<= muxenatomem;
		readmem_eoe					<= muxeoetomem;
		if(muxenatomem = '1') then
			readmem_addr_reg 		<= readmem_addr_reg + '1';
			if(muxeoetomem = '1')then
				readmem_eoeaddr_reg	<= readmem_addr_reg + '1';
			end if;			
		end if;
	else
		readmem_wren				<= '0';
		readmem_eoe					<= '0';
	end if;
end if;
end process;

end rtl;