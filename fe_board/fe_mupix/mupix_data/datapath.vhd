-- Mupix 8 data path
-- Sebastian Dittmeier, 08.4.2017
-- dittmeier@physi.uni-heidelberg.de
-- Changed by Marius Koeppel 18.07.2019
-- makoeppe@students.uni-mainz.de


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity data_path is
generic(
	NCHIPS 		: integer :=  4;
	NGX		: integer := 14;
	NLVDS 		: integer := 16;
	NSORTERINPUTS	: integer :=  4	--up to 4 LVDS links merge to one sorter
);
port (
	resets_n:		in reg32;
	resets: 		in reg32;
	slowclk:		in std_logic;
	clk125:			in std_logic;
	counter125:		in reg64;
	
	serial_data_in:		in std_logic_vector(NGX-1 downto 0);
	lvds_data_in:		in std_logic_vector(NLVDS-1 downto 0);
	
	clkext_out:		out std_logic_vector(NCHIPS-1 downto 0);
	
	writeregs:		in reg32array;
	regwritten:		in std_logic_vector(NREGISTERS-1 downto 0);

	readregs_slow: 		out reg32array;
	
	readmem_clk:		out std_logic;
	readmem_data:		out reg32;
	readmem_addr:		out readmemaddrtype;
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
constant use_LVDS  	: boolean := true;	-- false
constant use_SingleRO 	: boolean := true;	-- true
constant use_pseudodata	: boolean := true;	-- false

-- signals from Transceiver block (NGX)
signal rx_clkout_async		: std_logic_vector(NLVDS-1 downto 0);
signal rx_k_async		: std_logic_vector(NLVDS-1 downto 0);
signal rx_data_async		: std_logic_vector(NLVDS*8-1 downto 0);

signal rx_freqlocked_out	: std_logic_vector(NLVDS-1 downto 0);
signal rx_sync_out		: std_logic_vector(NLVDS-1 downto 0);
signal rx_err_out		: std_logic_vector(NLVDS-1 downto 0);
signal rx_disperr_out		: std_logic_vector(NLVDS-1 downto 0);
signal rx_pll_locked_out	: std_logic_vector(NLVDS-1 downto 0);
signal rx_patterndetect_out	: std_logic_vector(NLVDS-1 downto 0);
	
signal rx_runcounter		: links_reg32;
signal rx_errcounter		: links_reg32;

signal mux_gx_register		: std_logic_vector(1 downto 0);
signal mux_gx_lvds		: std_logic;
signal mux_lvds_clk		: std_logic_vector(NLVDS-1 downto 0);

-- signals from LVDS block (NLVDS)
signal lvds_clkout_async	: std_logic_vector(1 downto 0);
signal lvds_clkout_tofifo	: std_logic_vector(NLVDS-1 downto 0);
signal lvds_k_async		: std_logic_vector(NLVDS-1 downto 0);
signal lvds_data_async		: std_logic_vector(NLVDS*8-1 downto 0);

signal lvds_syncstatus		: std_logic_vector(NLVDS-1 downto 0);
signal lvds_pll_locked		: std_logic_vector(1 downto 0);
signal lvds_dpa_locked		: std_logic_vector(NLVDS-1 downto 0);
	
signal lvds_runcounter		: links_reg32;
signal lvds_errcounter		: links_reg32;	

-- signals from syncfifo to mux
signal rx_k_sync		: std_logic_vector(NLVDS-1 downto 0);
signal rx_data_sync		: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_data_valid_sync	: std_logic_vector(NLVDS-1 downto 0);
signal lvds_k_sync		: std_logic_vector(NLVDS-1 downto 0);
signal lvds_data_sync		: std_logic_vector(NLVDS*8-1 downto 0);
signal lvds_data_valid_sync	: std_logic_vector(NLVDS-1 downto 0);

-- signals after mux
signal rx_data			: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_k			: std_logic_vector(NLVDS-1 downto 0);
signal data_valid		: std_logic_vector(NLVDS-1 downto 0);

-- signals to the unpacker
signal rx_k_in			: std_logic_vector(NLVDS-1 downto 0);
signal rx_data_in		: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_syncstatus_in		: std_logic_vector(NLVDS-1 downto 0);
signal rx_syncstatus_nomask	: std_logic_vector(NLVDS-1 downto 0);

-- signals from the MP8 data generator, to muxed with data inputs
signal rx_data_gen		: std_logic_vector(NLVDS*8-1 downto 0);
signal rx_k_gen			: std_logic_vector(NLVDS-1 downto 0);
signal rx_syncstatus_gen	: std_logic_vector(NLVDS-1 downto 0);

signal gen_state_out		: std_logic_vector(NCHIPS*24-1 downto 0);
signal gen_linken		: std_logic_vector(NCHIPS*4-1 downto 0);
-- signal to mux between actual data and on-FPGA generated data
signal mux_fake_data		: std_logic := '0';

-- hits + flag to indicate a word as a hit, after unpacker
signal hits 			: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal hits_ena			: std_logic_vector(NLVDS-1 downto 0);
signal hits_i 			: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal hits_ena_i		: std_logic_vector(NLVDS-1 downto 0);

signal hits_i_mp8 		: std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
signal hits_ena_i_mp8 		: std_logic;
signal coarsecounters_i_mp8 	: std_logic_vector(COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_i_mp8	: std_logic;
signal unpack_errorcounter_mp8 	: reg32;
signal link_flag_i_mp8		: std_logic;

-- hits after gray-decoding
signal binhits_TS		: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena_TS		: std_logic_vector(NLVDS-1 downto 0);
signal binhits 			: std_logic_vector(NLVDS*UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena		: std_logic_vector(NLVDS-1 downto 0);
signal binhits_mp8 		: std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena_mp8		: std_logic;

-- hits to sorter
signal sorterhits 		: std_logic_vector(4*NSORTERINPUTS*UNPACKER_HITSIZE-1 downto 0);
signal sorterhits_ena		: std_logic_vector(4*NSORTERINPUTS-1 downto 0);
signal muxbinhit		: std_logic_vector(MHITSIZE-1 downto 0);
signal muxbinhitena		: std_logic;
signal muxcounterena		: std_logic;

-- flag to indicate link, after unpacker
signal link_flag 		: std_logic_vector(NLVDS-1 downto 0);
signal link_flag_i		: std_logic_vector(NLVDS-1 downto 0);

-- link flag is pipelined once because hits are gray decoded
signal link_flag_del 		: std_logic_vector(NLVDS-1 downto 0);

-- link flag is pipelined twice because hits are gray decoded
signal link_flag_reg 		: std_logic_vector(NLVDS-1 downto 0);

-- counter + flag to indicate word as a counter, after unpacker
signal coarsecounters 		: std_logic_vector(NLVDS*COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena 	: std_logic_vector(NLVDS-1 downto 0);
signal coarsecounters_i   	: std_logic_vector(NLVDS*COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_i	: std_logic_vector(NLVDS-1 downto 0);

-- counter is pipelined once because hits are gray decoded
signal coarsecounters_del 	: std_logic_vector(NLVDS *COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_del 	: std_logic_vector(NLVDS-1 downto 0);

-- counter is pipelined twice because hits are gray decoded
signal coarsecounters_reg 	: std_logic_vector(NLVDS *COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_reg 	: std_logic_vector(NLVDS-1 downto 0);

-- counter for sorter
signal sortercoarsecounters 	: std_logic_vector(4*NSORTERINPUTS*COARSECOUNTERSIZE-1 downto 0);
signal sortercoarsecounters_ena	: std_logic_vector(4*NSORTERINPUTS-1 downto 0);

-- muxed hits for single chip RO
signal single_hits 		: std_logic_vector(UNPACKER_HITSIZE-1 downto 0);
signal single_hits_ena 		: std_logic;
signal single_coarsecounters 	: std_logic_vector(COARSECOUNTERSIZE-1 downto 0);
signal single_coarsecounters_ena: std_logic;
signal single_link_flag 	: std_logic;							-- TODO: not yet connected anywhere
signal single_chipmarker	: std_logic_vector(7 downto 0);

-- this is the mux
signal single_res		: std_logic_vector(COARSECOUNTERSIZE+UNPACKER_HITSIZE+3-1 downto 0);
type single_mux_array is array (NLVDS-1 downto 0) of std_logic_vector(COARSECOUNTERSIZE+UNPACKER_HITSIZE+3-1 downto 0);
signal single_muxdata 		: single_mux_array;

-- signals between hit serializer and sortersignal 
signal hit_ser_out		: hit_array_t;
signal hit_ser_ena		: STD_LOGIC_VECTOR(NSORTERINPUTS-1 downto 0);
signal time_ser_out		: counter_array_t;
signal time_ser_ena		: STD_LOGIC_VECTOR(NSORTERINPUTS-1 downto 0);
signal hits_to_sorter		: hit_array_t;

-- signals for the hit counter (pre and post sorter)
signal sorter_out_hit		: std_logic;

-- RO mode ouputs
-- single chip RO
signal single2mem		: reg32;
signal single2memena		: std_logic;
signal single2memeoe		: std_logic;
-- single chip RO with zero suppression
signal singlezs2mem		: reg32;
signal singlezs2memena		: std_logic;
signal singlezs2memeoe		: std_logic;
-- telescope RO
signal telescope2mem		: reg32;
signal telescope2memena		: std_logic;
signal telescope2memeoe		: std_logic;
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
signal muxtomem			: reg32;
signal muxenatomem		: std_logic;
signal muxeoetomem		: std_logic;
signal readmem_addr_reg: readmemaddrtype;
signal readmem_eoeaddr_reg: readmemaddrtype;


-- histograms!
signal histo_komma_out				: std_logic_vector(47 downto 0);
signal histo_counter_out			: reg64array;
signal histo_hits_row				: chips_reg32;
signal histo_hits_col				: chips_reg32;
signal histo_hits_ts					: chips_reg32;
signal histo_hits_matrix			: chips_reg32;
signal histo_hits_multi				: chips_reg32;
signal histo_timediff				: chips_reg32;
signal histo_hitdiff					: chips_reg32;
signal histo_hitTSdiff				: chips_reg32;
signal histo_hitTSdiff_4mux1		: reg32;
signal histo_timediff_col			:	chips_histo_reg32;
signal histo_timediff_row			:	chips_histo_reg32;

-- hit counter pre- and postsort
--signal 	presort_hits	: 		reg64array;
--signal 	postsort_hits 	: 		reg64array;
signal 	presort_sum		: 		std_logic_vector(47 downto 0);
signal 	postsort_sum	: 		std_logic_vector(47 downto 0);
type	serializer_reg48_array is array (NSORTERINPUTS-1 downto 0) of std_logic_vector(47 downto 0);
signal	presort_per_serializer	: serializer_reg48_array;
signal presort_sum_r_0 : std_logic_vector(47 downto 0);
signal presort_sum_r_1 : std_logic_vector(47 downto 0);

--signal sync_fifo_mux_sel: std_logic_vector(2 downto 0);
-- mux for pre-post sort counters
signal sort_mux_sel		: std_logic_vector(2 downto 0);

--signal histo_mux_data0x	: 		reg32;
-- error signal output from unpacker
signal unpack_errorout		: links_reg32;
signal unpack_errorcounter	: links_reg32;
signal unpack_readycounter	: links_reg32;

-- muxing trigger fifo for RO modes
signal muxreadtrigfifo0			: std_logic;
signal muxreadtrigfifo1			: std_logic;
signal muxreadtrigfifo2			: std_logic;
signal muxreadtrigfifo3			: std_logic;
signal muxreadtrigfifo4			: std_logic;
signal muxreadtrigfifo5			: std_logic;
signal muxreadtrigfifo6			: std_logic;
signal muxreadtrigfifo7			: std_logic;

-- trigger fifo signals for RO modes
signal telescope_readtrigfifo		: std_logic;
signal telescope_trigfifoempty	: std_logic;
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
signal telescope_hitbusfifoempty	: std_logic;
signal allsinglezs2_readhitbusfifo		: std_logic;
signal allsinglezs2_hitbusfifoempty	: std_logic;

-- writeregisters are registered once to reduce long combinational paths
signal writeregs_reg:			reg32array;
signal regwritten_reg:		 	std_logic_vector(NREGISTERS-1 downto 0); 

--signal errcounter_identhit_rx			: reg32;	-- Identical hits after GX RX
--signal errout_identhit_rx				: reg32;	-- Identical hit data after GX RX
--signal errcounter_identhit_sync		: reg32;	-- Identical hits after sync fifos
--signal errout_identhit_sync			: reg32;	-- Identical hit data after sync fifos
--signal errcounter_identhit_unpack	: reg32;	-- Identical hits after unpacker
--signal errout_identhit_unpack			: reg32;	-- Identical hit data after unpacker
--signal errcounter_identhit_degray	: reg32;	-- Identical hits after gray decoding
--signal errout_identhit_degray			: reg32;	-- Identical hit data after gray decoding
--
----signal errcounter_identhit_rx_array			: chips_reg32;	-- Identical hits after GX RX
--signal errout_identhit_rx_array				: chips_reg32;	-- Identical hit data after GX RX
----signal errcounter_identhit_sync_array		: chips_reg32;	-- Identical hits after sync fifos
--signal errout_identhit_sync_array			: chips_reg32;	-- Identical hit data after sync fifos
----signal errcounter_identhit_unpack_array	: chips_reg32;	-- Identical hits after unpacker
--signal errout_identhit_unpack_array			: chips_reg32;	-- Identical hit data after unpacker
----signal errcounter_identhit_degray_array	: chips_reg32;	-- Identical hits after gray decoding
--signal errout_identhit_degray_array			: chips_reg32;	-- Identical hit data after gray decoding
--
--signal	err_identhit_rx		:  reg64;	-- Identical hits after GX RX
--signal	err_identhit_sync		:  reg64;	-- Identical hits after sync fifos
--signal	err_identhit_unpack	:  reg64;	-- Identical hits after unpacker
--signal	err_identhit_degray	:  reg64;		-- Identical hits after gray decoding
--
--signal errcounter_identhit_rx_add			: std_logic_vector(3 downto 0);
--signal errcounter_identhit_sync_add			: std_logic_vector(3 downto 0);
--signal errcounter_identhit_unpack_add		: std_logic_vector(3 downto 0);
--signal errcounter_identhit_degray_add		: std_logic_vector(3 downto 0);

-- hitsorter resets and debug signals
--signal		reset_readinithitsortmem_n 	:	std_logic;
--signal		reset_writeinithitsortmem_n	:	std_logic;

signal		cnt_hitswritten	:  reg64;
signal		cnt_hitsread		:  reg64;
signal		cnt_overflow		:  reg64;
signal		cnt_coarse			:  reg64;

--signal		errcounter_readwrite			:	reg32;	-- Reads during writes
--signal		errcounter_writedone			: 	reg32;	-- Writes after done
--signal		errcounter_zeroaddr			: 	reg32;	-- zero addresses
--signal		errcounter_hcnt_empty		: 	reg32;	-- Problems with hit counters, or empty blocks
--signal		errcounter_reset				: 	reg32;	-- Reset while writing or reading
--signal		errcounter_read_old			: 	reg32;	-- Reading while read, or old hits
--signal		errcounter_ignoredhits		: 	reg32;	-- Ignored incoming hits because we are still reading from this block
--signal		errcounter_ignoredblocks	: 	reg32;	-- Ignored complete blocks
--signal		errcounter_identhit_read	:  reg32;	-- Identical hits in read process
--signal		errout_identhit_read			:  reg32;	-- Identical hit data in read process
--signal		errcounter_identhit_write	:  reg32;	-- Identical hits in write process
--signal		errout_identhit_write		:  reg32;	-- Identical hit data in write process

--signal		hitsorter_debug				: hitsorter_debug_array;

-- some more debug signals
--signal rx_data_async_last				: reg32;
--signal rx_data_async_last2				: reg32;
--signal rx_k_async_last					: std_logic_vector(3 downto 0);
--signal rx_data_last						: reg32;
--signal rx_data_last2						: reg32;
--signal rx_k_last							: std_logic_vector(3 downto 0);
--signal hits_last							: std_logic_vector(NCHIPS*HITSIZE-1 downto 0);							
--signal binhits_last						: std_logic_vector(NCHIPS*HITSIZE-1 downto 0);
--signal errcounter_rx_rst				: std_logic;
--signal errcnt_unpack_rst				: std_logic;

signal readregs_muxed_gx 	: reg32array;
signal readregs_muxed_lvds : reg32array;

signal is_atlaspix : std_logic_vector(NLVDS-1 downto 0);
signal is_mp7		 : std_logic;


begin


writregs_clocking : process(clk125)
begin
	if(rising_edge(clk125))then
		for I in 63 downto 0 loop
			writeregs_reg(I)	<= writeregs(I);
			regwritten_reg(I)	<= regwritten(I);		
		end loop;
	end if;
end process writregs_clocking;


------------------------------------------------------------------------------------
---------------------- Transceiver part -------------------------------

-- We have to put two transceivers, because the pins are not adjacent
-- This breaks the geneic nature of the code, but then again, pins are 
-- never generic...

------- commented out for first tries
--
--errcounter_rx_rst <= resets_n(RESET_BIT_RECEIVER+0) or resets_n(RESET_BIT_RECEIVER+1) or resets_n(RESET_BIT_RECEIVER+2) or resets_n(RESET_BIT_RECEIVER+3);
--errcounter_rx_gen:
--for i in NCHIPS-1 downto 0 generate
--	process (resets_n(RESET_BIT_RECEIVER+i), rx_clkout(i))
--	begin
--		if (resets_n(RESET_BIT_RECEIVER+i) = '0') then
--			errcounter_identhit_rx_add(i)					<= '0';
--			errout_identhit_rx_array(i)					<= (others => '0');	
--			rx_data_async_last(8*(i+1)-1 downto 8*i)	<= (others => '0');
--			rx_data_async_last2(8*(i+1)-1 downto 8*i)	<= (others => '0');
--			rx_k_async_last(i)								<= '0';
--		elsif rising_edge(rx_clkout(i)) then
--			errcounter_identhit_rx_add(i) 	<= '0';
--			if rx_syncstatus(i) = '1' then
--				rx_data_async_last(8*(i+1)-1 downto 8*i)	<= rx_data_async(8*(i+1)-1 downto 8*i);
--				rx_data_async_last2(8*(i+1)-1 downto 8*i)	<= rx_data_async_last(8*(i+1)-1 downto 8*i);
--				rx_k_async_last(i)								<= rx_k_async(i);
--				if ( rx_k_async(i) = '0' and rx_k_async_last(i) = '0' ) then
--					if ( rx_data_async(8*(i+1)-1 downto 8*i) = rx_data_async_last(8*(i+1)-1 downto 8*i) ) then
--						errcounter_identhit_rx_add(i) 	<= '1';
--						errout_identhit_rx_array(i)		<= std_logic_vector(to_unsigned(i,4)) & "0000" & rx_data_async_last2(8*(i+1)-1 downto 8*i) & rx_data_async_last(8*(i+1)-1 downto 8*i) & rx_data_async(8*(i+1)-1 downto 8*i);
--					end if;				
--				end if;
--			end if;	
--		end if;
--	end process;
--end generate;
--
--process(errcounter_rx_rst, clk125)
--begin
--	if errcounter_rx_rst = '0' then
--		errcounter_identhit_rx	<= (others => '0');
--		errout_identhit_rx		<= (others => '0');
--	elsif rising_edge(clk125) then
--		errcounter_identhit_rx	<= errcounter_identhit_rx + errcounter_identhit_rx_add;
--		if 	errcounter_identhit_rx_add(0) = '1' then
--			errout_identhit_rx	<= errout_identhit_rx_array(0);
--		elsif	errcounter_identhit_rx_add(1) = '1' then
--			errout_identhit_rx	<= errout_identhit_rx_array(1);
--		elsif errcounter_identhit_rx_add(2) = '1' then
--			errout_identhit_rx	<= errout_identhit_rx_array(2);
--		elsif errcounter_identhit_rx_add(3) = '1' then
--			errout_identhit_rx	<= errout_identhit_rx_array(3);
--		end if;
--	end if;
--end process;


tb: transceiver_block 
generic map (
	NGX => NGX,
	NLVDS => NLVDS)
port map(
	reset_n					=> resets_n(RESET_RECEIVER_BIT),
	reset_cpu_n				=> resets_n(RESET_CPU_BIT),
	reset_n_errcnt			=> resets_n(RESET_ERRCOUNTER_BIT),
	slowclk					=> slowclk,
	clk125					=> clk125,
	serial_data_in			=> serial_data_in,
	
	clkext_out				=> clkext_out,

	rx_clkout				=> rx_clkout_async,
	rx_data_async_out		=> rx_data_async,
	rx_k_async_out			=> rx_k_async,
	
	rx_freqlocked_out		=> rx_freqlocked_out,
	rx_sync_out				=> rx_sync_out,
	rx_err_out				=> rx_err_out,
	rx_disperr_out			=> rx_disperr_out,
	rx_pll_locked_out		=> rx_pll_locked_out,
	rx_patterndetect_out	=> rx_patterndetect_out,
	
	rx_runcounter_out		=> rx_runcounter,
	rx_errorcounter_out	=> rx_errcounter,
	
	nios_address			=> writeregs_reg(NIOS_ADDRESS_REGISTER_W)(NIOS_ADDRESS_RANGE),
	nios_wr					=> writeregs_reg(NIOS_ADDRESS_REGISTER_W)(NIOS_WR_BIT),
	nios_rd					=> writeregs_reg(NIOS_ADDRESS_REGISTER_W)(NIOS_RD_BIT),
	nios_reg_written		=> regwritten_reg(NIOS_ADDRESS_REGISTER_W),
	nios_wrdata				=> writeregs_reg(NIOS_WRDATA_REGISTER_W),
	nios_rddata				=> readregs_slow(NIOS_RDDATA_REGISTER_R)
	
--	flash_tcm_address_out                       => flash_tcm_address_out,
--	flash_tcm_outputenable_n_out                => flash_tcm_outputenable_n_out,
--	flash_tcm_write_n_out                       => flash_tcm_write_n_out,
--	flash_tcm_data_out                          => flash_tcm_data_out,
--	flash_tcm_chipselect_n_out                  => flash_tcm_chipselect_n_out	
);


-- muxing between channels into the readregisters!
mux_gx_register <= writeregs(RECEIVER_REGISTER_W)(GX_REGISTER_MUX_RANGE);

gen_gx_reg:
For i in NCHIPS-1 downto 0 generate
	with mux_gx_register select readregs_muxed_gx(RECEIVER_RUNTIME_REGISTER_R+i) <=
		rx_runcounter(i*4) 	when "00",
		rx_runcounter(i*4+1) when "01",
		rx_runcounter(i*4+2)	when "10",
		rx_runcounter(i*4+3) when "11";	
		
	with mux_gx_register select readregs_muxed_gx(RECEIVER_ERRCOUNT_REGISTER_R+i) <=
		rx_errcounter(i*4) 	when "00",
		rx_errcounter(i*4+1) when "01",
		rx_errcounter(i*4+2)	when "10",
		rx_errcounter(i*4+3) when "11";			
end generate gen_gx_reg;

with mux_gx_register select readregs_slow(RECEIVER_REGISTER_R)(FREQLOCKED_RANGE) <=
		rx_freqlocked_out 	when "00",
		rx_pll_locked_out 	when "01",
		rx_patterndetect_out when "10",
		rx_freqlocked_out 	when others;		

with mux_gx_register select readregs_slow(RECEIVER_REGISTER_R)(SYNCSTATUS_RANGE) <=
		rx_sync_out 	when "00",
		rx_err_out 		when "01",
		rx_disperr_out when "10",
		rx_sync_out 	when others;	
		
---- Map registers
--readregs_slow(RECEIVER_REGISTER_R)(FREQLOCKED_RANGE)		<= rx_freqlocked_out;	-- important! first page!
--readregs_slow(RECEIVER_REGISTER_R)(D	PERR_RANGE)			<= rx_disperr_out;		-- less important -> mux this to 2nd and 3rd page
--readregs_slow(RECEIVER_REGISTER_R)(ERR_RANGE)				<= rx_err_out;				-- less important -> mux this to 2nd and 3rd page
--readregs_slow(RECEIVER_REGISTER_R)(PATTERNDETECT_RANGE)	<= rx_patterndetect_out;-- less important -> mux this to 2nd and 3rd page
--readregs_slow(RECEIVER_REGISTER_R)(PLL_LOCKED_RANGE)		<= rx_pll_locked_out;	-- less important -> mux this to 2nd and 3rd page
--readregs_slow(RECEIVER_REGISTER_R)(SYNCSTATUS_RANGE)		<= rx_sync_out;			-- important! first page!


------------------------------------------------------------------------------------
---------------------- LVDS Receiver part -------------------------------

bool_gen_lvds :
if use_LVDS generate

	lvds_block: receiver_block 
		generic map(
			NINPUT	=> NLVDS,
			NCHIPS	=> NCHIPS
		)
		port map(
			reset_n				=> resets_n(RESET_LVDS_BIT),
			reset_n_errcnt		=> resets_n(RESET_ERRCOUNTER_BIT),
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
	For i in NCHIPS-1 downto 0 generate
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

	readregs_slow(LVDS_REGISTER_R)(DPALOCKED_RANGE) 	<=	lvds_dpa_locked; 
	readregs_slow(LVDS_REGISTER_R)(RXREADY_RANGE) 		<=	lvds_syncstatus;
	--2bdefined! 
	readregs_slow(LVDS_REGISTER2_R)(LVDS_PLL_LOCKED_RANGE)	<= lvds_pll_locked;

end generate bool_gen_lvds;
	
		
--- --- --- --- --- --- --- --- -- --- --- --- --- --- ---
--- --- --- --- Muxing between GX and LVDS --- --- --- ---
-- but GX does not have all 16 channels, only 14, so we introduce two empty channels at MSB positions

-- Sync the four channels per chip to one clock --
sf_gx: syncfifos 
generic map(
	NCHIPS => NLVDS
)
port map(
	ready				=> rx_sync_out,
	clkout			=> clk125,
	reset_n			=> resets_n(RESET_RO_BIT),
	
	clkin				=> rx_clkout_async,
	datain			=> rx_data_async,
	kin				=> rx_k_async,
	
	dataout			=> rx_data_sync,
	kout				=> rx_k_sync,
	data_valid		=> rx_data_valid_sync
);

lvds_clkout_tofifo(15 downto 8) 	<= (others => lvds_clkout_async(1));
lvds_clkout_tofifo(7 downto 0) 	<= (others => lvds_clkout_async(0));


bool_gen_lvds_sf :
if use_LVDS generate

	sf_lvds: syncfifos 
	generic map(
		NCHIPS => NLVDS
	)
	port map(
		ready				=> lvds_syncstatus,
		clkout			=> clk125,
		reset_n			=> resets_n(RESET_RO_BIT),
		
		clkin				=> lvds_clkout_tofifo,
		datain			=> lvds_data_async,
		kin				=> lvds_k_async,
		
		dataout			=> lvds_data_sync,
		kout				=> lvds_k_sync,
		data_valid		=> lvds_data_valid_sync
	);
	
end generate bool_gen_lvds_sf;


mux_gx_lvds <= writeregs(RECEIVER_REGISTER_W)(GX_LVDS_MUX_BIT);
-- '0' selects GX receiver
-- '1' selects LVDS receiver

gen_runtime_error : 
for i in NCHIPS-1 downto 0 generate
	with mux_gx_lvds select readregs_slow(RECEIVER_RUNTIME_REGISTER_R+i) <=
		readregs_muxed_gx(RECEIVER_RUNTIME_REGISTER_R+i)		when '0',
		readregs_muxed_lvds(RECEIVER_RUNTIME_REGISTER_R+i)		when '1';
		
	with mux_gx_lvds select readregs_slow(RECEIVER_ERRCOUNT_REGISTER_R+i) <=
		readregs_muxed_gx(RECEIVER_ERRCOUNT_REGISTER_R+i)		when '0',
		readregs_muxed_lvds(RECEIVER_ERRCOUNT_REGISTER_R+i)	when '1';
end generate gen_runtime_error;

-- Here we hard disable links 1,2,3 for each chip in order to temporarily ease compilation
-- I removed that again (Seb)

-- mux data
with mux_gx_lvds select rx_data <=
		rx_data_sync 		when '0',
		lvds_data_sync 	when '1';
-- mux k
with mux_gx_lvds select rx_k <=
		rx_k_sync 			when '0',
		lvds_k_sync 		when '1';	

--rx_k(3 downto 1)  <= (others => '0');		
--		
--with mux_gx_lvds select rx_k(4) <=
--		rx_k_sync(4) 			when '0',
--		lvds_k_sync(4) 		when '1';	
--
--rx_k(7 downto 5)  <= (others => '0');	
--
--with mux_gx_lvds select rx_k(8) <=
--		rx_k_sync(8) 			when '0',
--		lvds_k_sync(8) 		when '1';	
--
--rx_k(11 downto 9)  <= (others => '0');	
--
--with mux_gx_lvds select rx_k(12) <=
--		rx_k_sync(12) 			when '0',
--		lvds_k_sync(12) 		when '1';	
--
--rx_k(15 downto 13)  <= (others => '0');	
		
		
-- mux data valid
with mux_gx_lvds select data_valid <=			
		rx_data_valid_sync		when '0',		
		lvds_data_valid_sync		when '1';

--data_valid(3 downto 1)  <= (others => '0');		
--		
--with mux_gx_lvds select data_valid(4) <=			
--		rx_data_valid_sync(4)		when '0',		
--		lvds_data_valid_sync(4)		when '1';
--		
--data_valid(7 downto 5)  <= (others => '0');		
--
--with mux_gx_lvds select data_valid(8) <=			
--		rx_data_valid_sync(8)		when '0',		
--		lvds_data_valid_sync(8)		when '1';		
--		
--data_valid(11 downto 9)  <= (others => '0');		
--				
--with mux_gx_lvds select data_valid(12) <=			
--		rx_data_valid_sync(12)		when '0',		
--		lvds_data_valid_sync(12)		when '1';	
--		
--data_valid(15 downto 13)  <= (others => '0');		
	
	
-- not touched below this except for NCHIPS --> NLVDS at some points!

------ commented out for first tries
--
--errcounter_syncfifo_gen:
--for i in NCHIPS-1 downto 0 generate
--	process (clk125, resets_n(RESET_BIT_RO))
--	begin
--		if (resets_n(RESET_BIT_RO) = '0') then
--			errcounter_identhit_sync_add(i)		<= '0';
--			errout_identhit_sync_array(i)			<= (others => '0');	
--			rx_data_last(8*(i+1)-1 downto 8*i)	<= (others => '0');
--			rx_data_last2(8*(i+1)-1 downto 8*i)	<= (others => '0');
--			rx_k_last(i)								<= '0';
--		elsif rising_edge(clk125) then
--			errcounter_identhit_sync_add(i) 	<= '0';
--			if data_valid(i) = '1' then
--				rx_data_last(8*(i+1)-1 downto 8*i)	<= rx_data(8*(i+1)-1 downto 8*i);
--				rx_data_last2(8*(i+1)-1 downto 8*i)	<= rx_data_last(8*(i+1)-1 downto 8*i);
--				rx_k_last(i)								<= rx_k(i);
--				if ( rx_k(i) = '0' and rx_k_last(i) = '0' ) then
--					if ( rx_data(8*(i+1)-1 downto 8*i) = rx_data_last(8*(i+1)-1 downto 8*i) ) then
--						errcounter_identhit_sync_add(i) 	<= '1';
--						errout_identhit_sync_array(i)		<= std_logic_vector(to_unsigned(i,4)) & "0000" & rx_data_last2(8*(i+1)-1 downto 8*i) & rx_data_last(8*(i+1)-1 downto 8*i) & rx_data(8*(i+1)-1 downto 8*i);
--					end if;				
--				end if;
--			end if;	
--		end if;
--	end process;
--end generate;
--
--process(clk125, resets_n(RESET_BIT_RO))
--begin
--	if resets_n(RESET_BIT_RO) = '0' then
--		errcounter_identhit_sync	<= (others => '0');
--		errout_identhit_sync		<= (others => '0');
--	elsif rising_edge(clk125) then
--		errcounter_identhit_sync	<= errcounter_identhit_sync + errcounter_identhit_sync_add;
--		if 	errcounter_identhit_sync_add(0) = '1' then
--			errout_identhit_sync	<= errout_identhit_sync_array(0);
--		elsif	errcounter_identhit_sync_add(1) = '1' then
--			errout_identhit_sync	<= errout_identhit_sync_array(1);
--		elsif errcounter_identhit_sync_add(2) = '1' then
--			errout_identhit_sync	<= errout_identhit_sync_array(2);
--		elsif errcounter_identhit_sync_add(3) = '1' then
--			errout_identhit_sync	<= errout_identhit_sync_array(3);
--		end if;
--	end if;
--end process;

------------------------------------------------------------------------------------
---------------- Generate data and multiplex between real and fake data ------------

gen_pseudodata_bool:
if use_pseudodata generate

mp8pseudodatagen:
FOR i in NCHIPS-1 downto 0 GENERATE
mp8datagen_i: pseudo_data_generator_mp8
	generic map (
		NumCOL				=> (32,48,48),
		NumROW				=> 200,
		MatrixSEL			=> ("00","01","10")
	)
	port map (
		reset_n				=> resets_n(RESET_DATA_GEN_BIT),
		syncres				=> resets(RESET_DATA_GEN_SYNCRES_BIT),															
		reset_pll			=> resets(RESET_PLL_BIT),
		clk125				=> clk125,									-- 125 MHz
		
		linken				=> gen_linken(i*4+3 downto i*4),
		
		-- talk to submatrices --> slowdown snap freq by factor slowdown
		slowdown				=> writeregs_reg(DATA_GEN_CHIP0_REGISTER_W+i)(SET_SLOWDOWN_RANGE),
		numhits				=> writeregs_reg(DATA_GEN_2_REGISTER_W)(SET_HIT_NUM_RANGE), -- same numhits for A,B,C
		
		-- readout state machine in digital part
		ckdivend				=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_CKDIVEND_RANGE),
		ckdivend2			=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_CKDIVEND2_RANGE),
		tsphase				=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_TSPHASE_RANGE),
		timerend				=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_TIMEREND_RANGE),
		slowdownend			=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_SLOWDOWNEND_RANGE),
		maxcycend			=> writeregs_reg(DATA_GEN_2_REGISTER_W)(DATA_GEN_MAXCYCEND_RANGE),	
		resetckdivend		=> writeregs_reg(DATA_GEN_2_REGISTER_W)(DATA_GEN_RESETCKDIVEND_RANGE),
		sendcounter			=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_SENDCOUNTER_BIT),
		linksel				=> writeregs_reg(DATA_GEN_2_REGISTER_W)(DATA_GEN_LINKSEL_RANGE),
		mode					=> writeregs_reg(DATA_GEN_REGISTER_W)(DATA_GEN_MODE_RANGE),
		
		dataout				=> rx_data_gen(i*32+31 downto i*32),
		kout					=> rx_k_gen(i*4+3 downto i*4),
		syncout				=> rx_syncstatus_gen(i*4+3 downto i*4),
		
		state_out			=> gen_state_out(i*24+23 downto i*24)
	);
	
END GENERATE;

readregs_slow(DATA_GEN_REGISTER_R) <= x"00" & gen_state_out(23 downto 0); -- display only CHIP0 here
gen_linken(3 downto 0) 		<= writeregs_reg(DATA_GEN_CHIP0_REGISTER_W)(DATA_GEN_LINKEN0_RANGE);
gen_linken(7 downto 4) 		<= writeregs_reg(DATA_GEN_CHIP1_REGISTER_W)(DATA_GEN_LINKEN1_RANGE);
gen_linken(11 downto 8) 	<= writeregs_reg(DATA_GEN_CHIP2_REGISTER_W)(DATA_GEN_LINKEN2_RANGE);
gen_linken(15 downto 12) 	<= writeregs_reg(DATA_GEN_CHIP3_REGISTER_W)(DATA_GEN_LINKEN3_RANGE);

end generate gen_pseudodata_bool;

-- clock this
	mux_process_datagenerator : process(clk125)
	begin
		if(rising_edge(clk125))then
			mux_fake_data <= writeregs_reg(DATA_GEN_REGISTER_W)(MUX_DATA_GEN_BIT);
			if(mux_fake_data = '0')then
				rx_data_in				<= rx_data;
				rx_k_in					<= rx_k;
				rx_syncstatus_nomask	<= data_valid;		
			else
				rx_data_in				<= rx_data_gen;	
				rx_k_in					<= rx_k_gen;	
				rx_syncstatus_nomask	<= rx_syncstatus_gen;				
			end if;
		end if;
	end process mux_process_datagenerator;

	rx_syncstatus_in <= rx_syncstatus_nomask and (not writeregs_reg(LINK_REGISTER_W)(LINK_MASK_TEL_RANGE));	-- masking unused channels

----------------------------------------------------------------------
-------------------------- NEW HISTOGRAMS-----------------------------	
----------------------------------------------------------------------
-----------------------all put in single top entity-------------------
-- AK: took out histograms to save logic resources :(
gen_histo_bool:
if use_Histogram generate

	i_histo_master : histo_master
	generic map(
		NCHIPS  => NLVDS,
		NHISTOS => 64,
		COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
		UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
	)
		port map(
			reset_n		=> resets_n(RESET_HISTOGRAMS_BIT),
			clk			=> clk125,
	-- ctrl		
			takedata		=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_TAKEDATA_BIT),
			zeromem		=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_ZEROMEM_BIT),
			written		=> regwritten_reg(HISTOGRAM_REGISTER_W),
			readaddr		=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_READADDR_RANGE),
			sel			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_SELECT_RANGE),
			hitmap_chipsel	=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_HITMAP_CHIPSEL_RANGE),	
	-- raw data	in
			datain		=> rx_data_in,
			kommain		=> rx_k_in,
	-- unpacker data in
			hit_in				=> binhits,
			hit_ena				=> binhits_ena,
			coarsecounter_in	=> coarsecounters_del,
			coarsecounter_ena	=> coarsecounters_ena_del,
			link_flag			=> link_flag_del,
	-- muxed hits in
			mux_hit_in			=> hit_ser_out,
			mux_hit_ena			=> hit_ser_ena,
			mux_coarsecounter_ena	=> time_ser_ena,
	-- output
			dataout_MSB		=> readregs_slow(HISTOGRAM_1_REGISTER_R),
			dataout_LSB		=> readregs_slow(HISTOGRAM_2_REGISTER_R)		
			);		
	
end generate gen_histo_bool;

----------------------------------------------------------------------------
------------------------------- SOME MORE HISTOGRAMS------------------------			
----------------------------------------------------------------------------
-------------- move them to histo_master if required!!!!--------------------
--
--
-- commented out for a start
--	histo_time_diff: mupix7_histo_hittime_coarse_diff  
--	port map(
--		rstn				=> resets_n(RESET_BIT_HISTOGRAMS),
--		clk				=> clk125,
--		takedata			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TAKEDATA),
--		zeromem			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_ZEROMEM),
--		binhittime		=> binhits((HITSIZE*i+TIMESTAMPSIZE-1) downto HITSIZE*i),
--		binhittime_ena	=> binhits_ena(i), 
--		coarsecnt		=> coarsecounters_del((COARSECOUNTERSIZE*(i)+BINCOUNTERSIZE-1) downto COARSECOUNTERSIZE*i),
--		coarsecnt_ena	=> coarsecounters_ena_del(i),
--		ckdivend			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_CKDIVEND_RANGE),		
--		readaddr			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TIMEDIFF_READADDR_RANGE),
--		written			=> regwritten_reg(HISTOGRAM_REGISTER_W),			
--		dataout_hit_coarse_diff		=> histo_timediff(i),
--		dataout_hit_hit_diff			=> histo_hitdiff(i)
--		);
--		
--	histo_hitTS_diff: mupix7_histo_hitTS_diff
--	port map(
--		rstn						=> resets_n(RESET_BIT_HISTOGRAMS),
--		clk						=> clk125,
--		takedata					=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TAKEDATA),
--		zeromem					=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_ZEROMEM),
--		binhittime				=> binhits((HITSIZE*i+TIMESTAMPSIZE-1) downto HITSIZE*i),
--		binhittime_ena			=> binhits_ena(i), 
--		coarsecnt_ena			=> coarsecounters_ena_del(i),
--	
--		readaddr					=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TIMEDIFF_READADDR_RANGE),
--		dataout_hitTS_diff	=> histo_hitTSdiff(i)
--		);
--
--	gencolumnhistos:
--	FOR j in 0 to 7 generate
--		histo_time_column : mupix7_histo_timediff_column  
--		generic map(
--			COL_START => HISTO_COL_START(j),
--			COL_END	 => HISTO_COL_END(j)
--		)
--			port map(
--				rstn				=> resets_n(RESET_BIT_HISTOGRAMS),
--				clk				=> clk125,
--				takedata			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TAKEDATA),
--				zeromem			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_ZEROMEM),
--				binhitcol		=> binhits((HITSIZE*i+13) downto (HITSIZE*i+8)),	
--				binhittime		=> binhits((HITSIZE*i+TIMESTAMPSIZE-1) downto HITSIZE*i),
--				binhittime_ena	=> binhits_ena(i), 
--				coarsecnt		=> coarsecounters_del((COARSECOUNTERSIZE*(i)+BINCOUNTERSIZE-1) downto COARSECOUNTERSIZE*i),
--				coarsecnt_ena	=> coarsecounters_ena_del(i),
--				written			=> regwritten_reg(HISTOGRAM_REGISTER_W),	
--				readaddr			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TIMEDIFF_READADDR_RANGE),
--				dataout_hit_coarse_diff		=> histo_timediff_col(j)(i)
--			);		
--		END GENERATE;
--		
--	genrowhistos:
--	FOR j in 0 to 7 generate
--		histo_time_row : mupix7_histo_timediff_column  
--		generic map(
--			COL_START => HISTO_ROW_START(j),
--			COL_END	 => HISTO_ROW_END(j)
--		)
--			port map(
--				rstn				=> resets_n(RESET_BIT_HISTOGRAMS),
--				clk				=> clk125,
--				takedata			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TAKEDATA),
--				zeromem			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_ZEROMEM),
--				binhitcol		=> binhits((HITSIZE*i+21) downto (HITSIZE*i+16)),	
--				binhittime		=> binhits((HITSIZE*i+TIMESTAMPSIZE-1) downto HITSIZE*i),
--				binhittime_ena	=> binhits_ena(i), 
--				coarsecnt		=> coarsecounters_del((COARSECOUNTERSIZE*(i)+BINCOUNTERSIZE-1) downto COARSECOUNTERSIZE*i),
--				coarsecnt_ena	=> coarsecounters_ena_del(i),
--				written			=> regwritten_reg(HISTOGRAM_REGISTER_W),	
--				readaddr			=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TIMEDIFF_READADDR_RANGE),
--				dataout_hit_coarse_diff		=> histo_timediff_row(j)(i)
--			);		
--		END GENERATE;	

--	histo_hitTS_4mux1_diff: mupix7_histo_hitTS_diff
--	port map(
--		rstn						=> resets_n(RESET_BIT_HISTOGRAMS),
--		clk						=> clk125,
--		
--		takedata					=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TAKEDATA),
--		zeromem					=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_ZEROMEM),
--		binhittime				=> muxbinhit(TIMESTAMPSIZE-1 downto 0),
--		binhittime_ena			=> muxbinhitena, 
--		coarsecnt_ena			=> muxcounterena,
--	
--		readaddr					=> writeregs_reg(HISTOGRAM_REGISTER_W)(HISTOGRAM_REGISTER_TIMEDIFF_READADDR_RANGE),
--		dataout_hitTS_diff	=> histo_hitTSdiff_4mux1
--		);		

------------------- Unpack the data ------------------------------------------------
-- Update for Mupix8!

gen_unpack_reg:
For i in NCHIPS-1 downto 0 generate
with mux_gx_register select readregs_slow(RECEIVER_UNPACKERRCOUNT_REGISTER_R+i) <=
		unpack_errorcounter(i*4) 	when "00",
		unpack_errorcounter(i*4+1) when "01",
		unpack_errorcounter(i*4+2)	when "10",
		unpack_errorcounter(i*4+3) when "11";	
end generate gen_unpack_reg;	


																		
--readregs_slow(RECEIVER_UNPACKREADYCOUNT_REGISTER_R)	<=	unpack_readycounter(7)(3 downto 0) & 		-- quick solution, should be muxable later
--																			unpack_readycounter(6)(3 downto 0) & 
--																			unpack_readycounter(5)(3 downto 0) & 
--																			unpack_readycounter(4)(3 downto 0) & 
--																			unpack_readycounter(3)(3 downto 0) & 
--																			unpack_readycounter(2)(3 downto 0) & 
--																			unpack_readycounter(1)(3 downto 0) & 
--																			unpack_readycounter(0)(3 downto 0) ;
																		
genunpack:
FOR i in NLVDS-1 downto 0 GENERATE	-- if logic utilization becomes a problem, build triple-unpacker only for the shared links

-- experimental: move to unpacker triple only for links 3, 7, 11 and 15
	MUXED_LINKS: if ((i mod 4) = 3) generate 

	unpacker_trips:data_unpacker_triple_new
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
	--		reset_out_n			=> resets_n(RESET_BIT_UNPACKER_OUT),
			clk					=> clk125,
			datain				=> rx_data_in(i*8+7 downto i*8),
			kin					=> rx_k_in(i),
			readyin				=> rx_syncstatus_in(i),
			is_shared			=> writeregs_reg(LINK_REGISTER_W)(i),
			is_atlaspix			=> is_atlaspix(i),
--			timerend				=> writeregs_reg(UNPACKER_CONFIG_REGISTER_W)(TIMEREND_RANGE),
			hit_out				=> hits_i(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena				=> hits_ena_i(i),
			coarsecounter		=> coarsecounters_i(COARSECOUNTERSIZE*(i+1)-1 downto COARSECOUNTERSIZE*i),
			coarsecounter_ena	=> coarsecounters_ena_i(i),
			link_flag			=> link_flag_i(i),
			errorcounter		=> unpack_errorcounter(i)
--			error_out			=> unpack_errorout(i),
--			readycounter		=> unpack_readycounter(i)
		);
		
		degray_trip : hit_ts_conversion 
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125, 
			invert_TS			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_INVERT_BIT),
			invert_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT),
			gray_TS				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			gray_TS2				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_in			=> hits_ena(i),
			hit_out				=> binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_out			=> binhits_ena(i)
			);
		
		
	end generate MUXED_LINKS;	
		
-- these here are definitely single links

--	gen_check_mp7_not:
--	if not use_DUT_MuPix7 generate
--	
--		SINGLE_LINKS: if ((i mod 4) /= 3) generate 
--		
--		unpacker:data_unpacker_new
--		generic map(
--			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
--			UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
--		)
--		port map(
--			reset_n				=> resets_n(RESET_RO_BIT),
--	--		reset_out_n			=> resets_n(RESET_BIT_UNPACKER_OUT),
--			clk					=> clk125,
--			datain				=> rx_data_in(i*8+7 downto i*8),
--			kin					=> rx_k_in(i),
--			readyin				=> rx_syncstatus_in(i),
--			is_atlaspix			=> is_atlaspix(i),
--			hit_out				=> hits_i((i+1)*UNPACKER_HITSIZE-1 downto i*UNPACKER_HITSIZE),
--			hit_ena				=> hits_ena_i(i),
--			coarsecounter		=> coarsecounters_i((i+1)*COARSECOUNTERSIZE-1 downto i*COARSECOUNTERSIZE),
--			coarsecounter_ena	=> coarsecounters_ena_i(i),
--			link_flag			=> link_flag_i(i),
--			errorcounter		=> unpack_errorcounter(i)
--			);	
--		
--		end generate SINGLE_LINKS;	
--		
--	end generate gen_check_mp7_not;
	
--	gen_check_mp7:
--	if use_DUT_MuPix7 generate
	
	SINGLE_LINKS: if ((i mod 4) /= 3 ) and (i /= 4) generate 
		
		unpacker_single:data_unpacker_new
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
	--		reset_out_n			=> resets_n(RESET_BIT_UNPACKER_OUT),
			clk					=> clk125,
			datain				=> rx_data_in(i*8+7 downto i*8),
			kin					=> rx_k_in(i),
			readyin				=> rx_syncstatus_in(i),
			is_atlaspix			=> is_atlaspix(i),
			hit_out				=> hits_i((i+1)*UNPACKER_HITSIZE-1 downto i*UNPACKER_HITSIZE),
			hit_ena				=> hits_ena_i(i),
			coarsecounter		=> coarsecounters_i((i+1)*COARSECOUNTERSIZE-1 downto i*COARSECOUNTERSIZE),
			coarsecounter_ena	=> coarsecounters_ena_i(i),
			link_flag			=> link_flag_i(i),
			errorcounter		=> unpack_errorcounter(i)
		);	
		
		degray_single : hit_ts_conversion 
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125, 
			invert_TS			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_INVERT_BIT),
			invert_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT),
			gray_TS				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			gray_TS2				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_in			=> hits_ena(i),
			hit_out				=> binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_out			=> binhits_ena(i)
			);
		
	end generate SINGLE_LINKS;	
		
	
	DUT_MuPix7_or_MuPix8: if (i = 4) generate 
		
		unpacker_single:data_unpacker_new
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
	--		reset_out_n			=> resets_n(RESET_BIT_UNPACKER_OUT),
			clk					=> clk125,
			datain				=> rx_data_in(i*8+7 downto i*8),
			kin					=> rx_k_in(i),
			readyin				=> rx_syncstatus_in(i),
			is_atlaspix			=> is_atlaspix(i),
			hit_out				=> hits_i_mp8,
			hit_ena				=> hits_ena_i_mp8,
			coarsecounter		=> coarsecounters_i_mp8,
			coarsecounter_ena	=> coarsecounters_ena_i_mp8,
			link_flag			=> link_flag_i_mp8,
			errorcounter		=> unpack_errorcounter_mp8
		);	
	
		unpacker_mp7: data_unpacker_mp7 
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE	=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125,
			datain				=> rx_data_in(i*8+7 downto i*8),
			kin					=> rx_k_in(i),
			readyin				=> rx_syncstatus_in(i),
			hit_out				=> hits_i_mp7,
			hit_ena				=> hits_ena_i_mp7,
			coarsecounter		=> coarsecounters_i_mp7,
			coarsecounter_ena	=> coarsecounters_ena_i_mp7,
			errorcounter		=> unpack_errorcounter_mp7
			--error_out			: OUT STD_LOGIC_VECTOR(31 downto 0)
			);
			
			link_flag_i_mp7 <= coarsecounters_ena_i_mp7; -- simply use coarsecounter flag as indicator that there is a new block for MuPix7.
				
		with is_mp7 select hits_i((i+1)*UNPACKER_HITSIZE-1 downto i*UNPACKER_HITSIZE) <= 
			hits_i_mp8	when '0',							-- mupix8 or ATLASpix
			hits_i_mp7 	when '1';							-- mupix7
												
		with is_mp7 select hits_ena_i(i) <= 
			hits_ena_i_mp8	when '0',						-- mupix8 or ATLASpix
			hits_ena_i_mp7 	when '1';					-- mupix7
	
		with is_mp7 select coarsecounters_i((i+1)*COARSECOUNTERSIZE-1 downto i*COARSECOUNTERSIZE) <= 
			coarsecounters_i_mp8	when '0',				-- mupix8 or ATLASpix
			coarsecounters_i_mp7 	when '1';			-- mupix7
					
		with is_mp7 select coarsecounters_ena_i(i) <= 
			coarsecounters_ena_i_mp8	when '0',		-- mupix8 or ATLASpix
			coarsecounters_ena_i_mp7 	when '1';		-- mupix7
		
		with is_mp7 select link_flag_i(i) <= 
			link_flag_i_mp8	when '0',					-- mupix8 or ATLASpix
			link_flag_i_mp7 	when '1';					-- mupix7
			
		with is_mp7 select unpack_errorcounter(i) <= 
			unpack_errorcounter_mp8	when '0',			-- mupix8 or ATLASpix
			unpack_errorcounter_mp7 	when '1';		-- mupix7
			
		-- we add two bits of the coarsecounter to the timestamp of the MuPix7 hits to increase the sorter efficiency in operation together with MuPix8
		with is_mp7 select binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i) <= 
			binhits_mp8		when '0',		-- mupix8 or ATLASpix
			binhits_mp7(UNPACKER_HITSIZE-1 downto TIMESTAMPSIZE_MPX8) & coarsecounters_i_mp7(TIMESTAMPSIZE_MPX8-1 downto TIMESTAMPSIZE_MP7) & binhits_mp7(TIMESTAMPSIZE_MP7-1 downto 0) 	when '1';		-- mupix7
			
		with is_mp7 select binhits_ena(i) <= 
			binhits_ena_mp8	when '0',			-- mupix8 or ATLASpix
			binhits_ena_mp7 	when '1';			-- mupix7

			
		degray_mp8 : hit_ts_conversion 
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125, 
			invert_TS			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_INVERT_BIT),
			invert_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT),
			gray_TS				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			gray_TS2				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_in			=> hits_ena(i),
			hit_out				=> binhits_mp8, --(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_out			=> binhits_ena_mp8--(i)
			);	
				
		-- if we want to gray-decode MP7 timestamps, they have to be inverted first
		-- hence we hard-couple bit inversion to gray-decoding for MP7
		degray_mp7 : hit_ts_conversion 
		generic map(
			TS_SIZE => TIMESTAMPSIZE_MP7
			)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125, 
			invert_TS			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			invert_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT),
			gray_TS				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			gray_TS2				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_in			=> hits_ena(i),
			hit_out				=> binhits_mp7,--(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_out			=> binhits_ena_mp7--(i)
			);			
				
	end generate DUT_MuPix7_or_MuPix8;
		
--	end generate gen_check_mp7;

	is_atlaspix <= writeregs_reg(RO_MODE_REGISTER_W)(RO_IS_ATLASPIX_RANGE);	
	is_mp7		<= writeregs_reg(RO_MODE_REGISTER_W)(DUT_IS_MP7_BIT);

-- Seb: Conversion to binary in one cycle for TS and Charge? might save some registers.		
		
--	----------------------------------------------------------------
--	-- HIT TIMESTAMP CONVERSION FROM INVERTED(GRAY) TO BINARY
--	----------------------------------------------------------------
--	degray:hit_gray_to_binary 
--		port map(
--			reset_n				=> resets_n(RESET_RO_BIT),
--			clk					=> clk125, 
--			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
--			hit_ena_in			=> hits_ena(i),
--			hit_out				=> binhits_TS(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
--			hit_ena_out			=> binhits_ena_TS(i)
--		);		
--	
--	----------------------------------------------------------------
--	-- CHARGE CONVERSION FROM INVERTED(GRAY) TO BINARY
--	----------------------------------------------------------------
--	degray_TOT:hit_gray_to_binary_TOT 
--		port map(
--			reset_n				=> resets_n(RESET_RO_BIT),
--			clk					=> clk125, 
--			hit_in				=> binhits_TS(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
--			hit_ena_in			=> binhits_ena_TS(i),
--			hit_out				=> binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
--			hit_ena_out			=> binhits_ena(i)
--		);		


END GENERATE genunpack;

-- register output from data unpacker because of timing issues
process (clk125, resets_n(RESET_RO_BIT))
begin
	if resets_n(RESET_RO_BIT)='0' then
		hits						<= (others => '0');
		hits_ena					<= (others => '0');
		coarsecounters			<= (others => '0');
		coarsecounters_ena	<= (others => '0');
		link_flag				<= (others => '0');
	elsif rising_edge(clk125) then
		hits						<= hits_i;
		hits_ena					<= hits_ena_i;
		coarsecounters			<= coarsecounters_i;
		coarsecounters_ena	<= coarsecounters_ena_i;
		link_flag				<= link_flag_i;	
	end if;
end process;

	-- delay cc by one cycle to be in line with hit
	-- Seb: new degray - back to one
	process(clk125)
	begin
	if(clk125'event and clk125 = '1') then
--		coarsecounters_reg			<= coarsecounters;
--		coarsecounters_ena_reg		<= coarsecounters_ena;
--		link_flag_reg					<= link_flag;
--		coarsecounters_del			<= coarsecounters_reg;
--		coarsecounters_ena_del		<= coarsecounters_ena_reg;
--		link_flag_del					<= link_flag_reg;
		coarsecounters_del			<= coarsecounters;
		coarsecounters_ena_del		<= coarsecounters_ena;
		link_flag_del					<= link_flag;
	end if;
	end process;	

--errcnt_unpack_rst <= resets_n(RESET_BIT_RO);--nand resets_n(RESET_BIT_UNPACKER_OUT);
--errcounter_unpack_gen:
--for i in NCHIPS-1 downto 0 generate
--	-- identical hits after unpacker/gray decoding
--	process (clk125, errcnt_unpack_rst)
--	begin
--		if (errcnt_unpack_rst = '0') then
--			-- unpacker
--			errcounter_identhit_unpack_add(i)					<= '0';
--			errout_identhit_unpack_array(i)						<= (others => '0');	
--			hits_last(HITSIZE*(i+1)-1 downto HITSIZE*i)		<= (others => '0');
--			-- degray
--			errcounter_identhit_degray_add(i)					<= '0';
--			errout_identhit_degray_array(i)						<= (others => '0');	
--			binhits_last(HITSIZE*(i+1)-1 downto HITSIZE*i)	<= (others => '0');
--		elsif rising_edge(clk125) then
--			-- unpacker
--			errcounter_identhit_unpack_add(i)	<= '0';
--			if ( hits_ena(i) = '1' and coarsecounters_ena(i) = '0' ) then
--				hits_last(HITSIZE*(i+1)-1 downto HITSIZE*i)	<= hits(HITSIZE*(i+1)-1 downto HITSIZE*i);
--				if ( hits_last(HITSIZE*(i+1)-1 downto HITSIZE*i) = hits(HITSIZE*(i+1)-1 downto HITSIZE*i) ) then
--					errcounter_identhit_unpack_add(i)	<= '1';
--					errout_identhit_unpack_array(i)		<= std_logic_vector(to_unsigned(i,4)) & "0000" & hits(HITSIZE*(i+1)-1 downto HITSIZE*i);			
--				end if;
--			end if;
--			-- degray
--			errcounter_identhit_degray_add(i)	<= '0';
--			if ( binhits_ena(i) = '1' and coarsecounters_ena_del(i) = '0' ) then
--				binhits_last(HITSIZE*(i+1)-1 downto HITSIZE*i)	<= binhits(HITSIZE*(i+1)-1 downto HITSIZE*i);
--				if ( binhits_last(HITSIZE*(i+1)-1 downto HITSIZE*i) = binhits(HITSIZE*(i+1)-1 downto HITSIZE*i) ) then
--					errcounter_identhit_degray_add(i)	<= '1';
--					errout_identhit_degray_array(i)		<= std_logic_vector(to_unsigned(i,4)) & "0000" & binhits(HITSIZE*(i+1)-1 downto HITSIZE*i);			
--				end if;
--			end if;					
--		end if;
--	end process;
--end generate;
--
--process(clk125, errcnt_unpack_rst)
--begin
--	if errcnt_unpack_rst = '0' then
--		-- unpacker
--		errcounter_identhit_unpack	<= (others => '0');
--		errout_identhit_unpack		<= (others => '0');
--		-- degray
--		errcounter_identhit_degray	<= (others => '0');
--		errout_identhit_degray		<= (others => '0');
--	elsif rising_edge(clk125) then
--		-- unpacker
--		errcounter_identhit_unpack	<= errcounter_identhit_unpack + errcounter_identhit_unpack_add;
--		if 	errcounter_identhit_unpack_add(0) = '1' then
--			errout_identhit_unpack	<= errout_identhit_unpack_array(0);
--		elsif	errcounter_identhit_unpack_add(1) = '1' then
--			errout_identhit_unpack	<= errout_identhit_unpack_array(1);
--		elsif errcounter_identhit_unpack_add(2) = '1' then
--			errout_identhit_unpack	<= errout_identhit_unpack_array(2);
--		elsif errcounter_identhit_unpack_add(3) = '1' then
--			errout_identhit_unpack	<= errout_identhit_unpack_array(3);
--		end if;
--		-- degray
--		errcounter_identhit_degray	<= errcounter_identhit_degray + errcounter_identhit_degray_add;
--		if 	errcounter_identhit_degray_add(0) = '1' then
--			errout_identhit_degray	<= errout_identhit_degray_array(0);
--		elsif	errcounter_identhit_degray_add(1) = '1' then
--			errout_identhit_degray	<= errout_identhit_degray_array(1);
--		elsif errcounter_identhit_degray_add(2) = '1' then
--			errout_identhit_degray	<= errout_identhit_degray_array(2);
--		elsif errcounter_identhit_degray_add(3) = '1' then
--			errout_identhit_degray	<= errout_identhit_degray_array(3);
--		end if;
--	end if;
--end process;
--
--	
--unpack_mux : memorymux
--PORT MAP
--	(
--		clock			=> clk125,
--		data0x		=> unpack_errorout(0),
--		data1x		=> unpack_errorout(1),
--		data2x		=> unpack_errorout(2),
--		data3x		=> unpack_errorout(3),
--		data4x		=> fifo_underflow(0),
--		data5x		=> fifo_underflow(1),
--		data6x		=> fifo_underflow(2),
--		data7x		=> fifo_underflow(3),
--		sel			=> sort_mux_sel,
--		result		=> readregs_slow(RECEIVER_UNPACKERROR_REGISTER_R)
--	);	
--	
sort_mux_sel	<= writeregs_reg(DEBUG_CHIP_SELECT_REGISTER_W)(CHIP_SELECT_RANGE);	

------------------------------------------------------------------------------------

------------------ Multiplex Chips for single chip RO mode -------------------------
--apply to all links!!!
gen_singleRO_bool:
if use_SingleRO generate

	gen_singlemux: 
	for i in 0 to NLVDS-1 generate
		single_muxdata(i) <= link_flag_del(i) & binhits_ena(i) & binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i) & coarsecounters_ena_del(i) & coarsecounters_del(COARSECOUNTERSIZE*(i+1)-1 downto COARSECOUNTERSIZE*i);

	end generate;
		
	singlechipmux: singlemux 
		PORT MAP
		(
			clock			=> clk125,
			data0x		=> single_muxdata(0),
			data1x		=> single_muxdata(1),
			data2x		=> single_muxdata(2),
			data3x		=> single_muxdata(3),
			data4x		=> single_muxdata(4),
			data5x		=> single_muxdata(5),
			data6x		=> single_muxdata(6),
			data7x		=> single_muxdata(7),
			data8x		=> single_muxdata(8),
			data9x		=> single_muxdata(9),	
			data10x		=> single_muxdata(10),
			data11x		=> single_muxdata(11),
			data12x		=> single_muxdata(12),
			data13x		=> single_muxdata(13),
			data14x		=> single_muxdata(14),
			data15x		=> single_muxdata(15),		
			sel			=> writeregs_reg(RO_MODE_REGISTER_W)(RO_MODE_SINGLECHIP_RANGE),
			result		=> single_res		
		);	
		
	 single_coarsecounters 		<= single_res(COARSECOUNTERSIZE-1 downto 0);
	 single_coarsecounters_ena <= single_res(COARSECOUNTERSIZE);
	 single_hits 					<= single_res(COARSECOUNTERSIZE+UNPACKER_HITSIZE DOWNTO COARSECOUNTERSIZE+1);
	 single_hits_ena 				<= single_res(COARSECOUNTERSIZE+UNPACKER_HITSIZE+1);
	 single_link_flag				<= single_res(COARSECOUNTERSIZE+UNPACKER_HITSIZE+2);
	------------------------------------------------------------------------------------
	-- use chosen link number as chipmarker
	 single_chipmarker 			<= x"0" & writeregs_reg(RO_MODE_REGISTER_W)(RO_MODE_SINGLECHIP_RANGE);
	------------------------ Single chip RO mode ---------------------------------------
	singlero: singlechip_ro 
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE				=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125,
			counter125			=> counter125,
			link_flag			=> single_link_flag,
			hit_in				=> single_hits,
			hit_ena				=> single_hits_ena,
			coarsecounter		=> single_coarsecounters,
			coarsecounter_ena	=> single_coarsecounters_ena,
			chip_marker			=> single_chipmarker,--writeregs_reg(RO_CHIPMARKER_REGISTER_W)(RO_CHIPMARKER_A_FRONT_RANGE),		
			tomemdata			=> single2mem,
			tomemena				=> single2memena,
			tomemeoe				=> single2memeoe
			);
	------------------------------------------------------------------------------------

	------------------------ Single chip RO mode, zero suppressed ----------------------
	singlerozs: singlechip_ro_zerosupressed 
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE				=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> resets_n(RESET_RO_BIT),
			clk					=> clk125,
			counter125			=> counter125,
			link_flag			=> single_link_flag,
			hit_in				=> single_hits,
			hit_ena				=> single_hits_ena,
			coarsecounter		=> single_coarsecounters,
			coarsecounter_ena	=> single_coarsecounters_ena,
			chip_marker			=> single_chipmarker,--writeregs_reg(RO_CHIPMARKER_REGISTER_W)(RO_CHIPMARKER_A_FRONT_RANGE),	
			prescale				=> writeregs_reg(RO_PRESCALER_REGISTER_W)(RO_PRESCALER_RANGE),
			tomemdata			=> singlezs2mem,
			tomemena				=> singlezs2memena,
			tomemeoe				=> singlezs2memeoe
			);
end generate gen_singleRO_bool;
		
---------------------------------------------------------------------------------------------------		

------------------------ For all chips: Single chip RO mode, zero suppressed ----------------------
-- now we also try to include the muxed links!

gen_multi_bool:
if use_MultiRO generate

	multirozs : multichip_ro_zerosupressed
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			HITSIZE				=> UNPACKER_HITSIZE,
			NCHIPS 				=> NLVDS	
		)
		port map(
			clk						=> clk125,												
			reset_n					=> resets_n(RESET_RO_BIT),		
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
			hbfifoempty				=> allsinglezs2_hitbusfifoempty	
			);	

end generate gen_multi_bool;
		
------------------------------------------------------------------------------------
-- adapt for MuPix8
------------------------- Sorted RO mode for up to four chips ----------------------

--coarsecounters_bin	<= coarsecounters_del(COARSECOUNTERSIZE*4-1 downto COARSECOUNTERSIZE*3 + TIMESTAMPSIZE) &
--								coarsecounters_del(COARSECOUNTERSIZE*3-1 downto COARSECOUNTERSIZE*2 + TIMESTAMPSIZE) &
--								coarsecounters_del(COARSECOUNTERSIZE*2-1 downto COARSECOUNTERSIZE*1 + TIMESTAMPSIZE) &
--								coarsecounters_del(COARSECOUNTERSIZE*1-1 downto COARSECOUNTERSIZE*0 + TIMESTAMPSIZE);
--process (clk125)
--begin
--if(rising_edge(clk125))then
----	telescope_hits			<= binhits;
----	telescope_hits_ena	<= binhits_ena;
----	coarsecounters_bin	<= coarsecounters_del(COARSECOUNTERSIZE*16-1 downto COARSECOUNTERSIZE*15 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*15-1 downto COARSECOUNTERSIZE*14 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*14-1 downto COARSECOUNTERSIZE*13 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*13-1 downto COARSECOUNTERSIZE*12 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*12-1 downto COARSECOUNTERSIZE*11 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*11-1 downto COARSECOUNTERSIZE*10 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*10-1 downto COARSECOUNTERSIZE*9 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*9-1 downto COARSECOUNTERSIZE*8 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*8-1 downto COARSECOUNTERSIZE*7 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*7-1 downto COARSECOUNTERSIZE*6 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*6-1 downto COARSECOUNTERSIZE*5 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*5-1 downto COARSECOUNTERSIZE*4 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*4-1 downto COARSECOUNTERSIZE*3 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*3-1 downto COARSECOUNTERSIZE*2 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*2-1 downto COARSECOUNTERSIZE*1 + TIMESTAMPSIZE) &
----									coarsecounters_del(COARSECOUNTERSIZE*1-1 downto COARSECOUNTERSIZE*0 + TIMESTAMPSIZE);
----	coarsecounters_bin_ena	<= coarsecounters_ena_del;
--	link_flag_bin				<= link_flag_del;
--end if;
--end process;								
--process(clk125)
--begin
--	if(rising_edge(clk125)) then
--		if(regwritten_reg(HITSORTER_INIT_REGISTER_W)='1')then
--			reset_readinithitsortmem_n 	<= not writeregs_reg(HITSORTER_INIT_REGISTER_W)(READINITHITSORTMEM_BIT);
--			reset_writeinithitsortmem_n 	<= not writeregs_reg(HITSORTER_INIT_REGISTER_W)(WRITEINITHITSORTMEM_BIT);			
--		end if;
--	end if;
--end process;

------------------------------------------
-- hit serializer and sorter
------------------------------------------
--sorterhits						<= sorterhits_diff & binhits;
--sorterhits_ena					<= sorter_ena_diff & binhits_ena;
--sortercoarsecounters			<= sortercoarsecounters_diff & coarsecounters_del;
--sortercoarsecounters_ena	<= sorter_ena_diff & coarsecounters_ena_del;

---- Seb: there are no differences in lvds links and sorter inputs --OLD
--sorterhits						<= binhits;
--sorterhits_ena					<= binhits_ena;
--sortercoarsecounters			<= coarsecounters_del;
--sortercoarsecounters_ena	<= coarsecounters_ena_del;

--reduced sorter
gen_sorterhits:	-- NSORTERINPUTS = 1, does not work properly otherwise! -- only links A
for i in 0 to NSORTERINPUTS-1 generate
	sorterhits((4*i+1)*UNPACKER_HITSIZE-1 downto (4*i)*UNPACKER_HITSIZE)		<= binhits((16*i+1)*UNPACKER_HITSIZE-1 downto (16*i)*UNPACKER_HITSIZE);			-- link A of 1st and 5th chip
	sorterhits((4*i+2)*UNPACKER_HITSIZE-1 downto (4*i+1)*UNPACKER_HITSIZE)	<= binhits((16*i+5)*UNPACKER_HITSIZE-1 downto (16*i+4)*UNPACKER_HITSIZE);		-- link A of 2nd and 6th chip
	sorterhits((4*i+3)*UNPACKER_HITSIZE-1 downto (4*i+2)*UNPACKER_HITSIZE)	<= binhits((16*i+9)*UNPACKER_HITSIZE-1 downto (16*i+8)*UNPACKER_HITSIZE);		-- link A of 3rd and 7th chip
	sorterhits((4*i+4)*UNPACKER_HITSIZE-1 downto (4*i+3)*UNPACKER_HITSIZE)	<= binhits((16*i+13)*UNPACKER_HITSIZE-1 downto (16*i+12)*UNPACKER_HITSIZE);	-- link A of 4th and 8th chip
	sorterhits_ena(4*i)			<= binhits_ena(16*i);	
	sorterhits_ena(4*i+1)		<= binhits_ena(16*i+4);	
	sorterhits_ena(4*i+2)		<= binhits_ena(16*i+8);
	sorterhits_ena(4*i+3)		<= binhits_ena(16*i+12);
	sortercoarsecounters((4*i+1)*COARSECOUNTERSIZE-1 downto (4*i)*COARSECOUNTERSIZE)				<= coarsecounters_del((16*i+1)*COARSECOUNTERSIZE-1 downto (16*i)*COARSECOUNTERSIZE);
	sortercoarsecounters((4*i+2)*COARSECOUNTERSIZE-1 downto (4*i+1)*COARSECOUNTERSIZE)			<= coarsecounters_del((16*i+5)*COARSECOUNTERSIZE-1 downto (16*i+4)*COARSECOUNTERSIZE);
	sortercoarsecounters((4*i+3)*COARSECOUNTERSIZE-1 downto (4*i+2)*COARSECOUNTERSIZE)			<= coarsecounters_del((16*i+9)*COARSECOUNTERSIZE-1 downto (16*i+8)*COARSECOUNTERSIZE);
	sortercoarsecounters((4*i+4)*COARSECOUNTERSIZE-1 downto (4*i+3)*COARSECOUNTERSIZE)			<= coarsecounters_del((16*i+13)*COARSECOUNTERSIZE-1 downto (16*i+12)*COARSECOUNTERSIZE);	
	sortercoarsecounters_ena(4*i)			<= coarsecounters_ena_del(16*i);	
	sortercoarsecounters_ena(4*i+1)		<= coarsecounters_ena_del(16*i+4);	
	sortercoarsecounters_ena(4*i+2)		<= coarsecounters_ena_del(16*i+8);
	sortercoarsecounters_ena(4*i+3)		<= coarsecounters_ena_del(16*i+12);
end generate;



gen_hitser:	-- the 4 MSB of hit_in and time_in encode the link number as chip marker
for i in 0 to NSORTERINPUTS-1 generate
	hit_ser: hit_serializer
	generic map(
		SERIALIZER_HITSIZE 	=> UNPACKER_HITSIZE+4,	-- 4 additional bits to encode the receiver
		SERBINCOUNTERSIZE		=> BINCOUNTERSIZE+4,
		SERHITSIZE				=> 40
	)
	port map(
		reset_n				=> resets_n(RESET_RO_BIT),
		clk					=> clk125,
		hit_in1				=> std_logic_vector(to_unsigned(4*i  ,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+1)-1 downto UNPACKER_HITSIZE*4*i),
		hit_ena1				=> sorterhits_ena(4*i),
		hit_in2				=> std_logic_vector(to_unsigned(4*i+1,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+2)-1 downto UNPACKER_HITSIZE*(4*i+1)),
		hit_ena2				=> sorterhits_ena(4*i+1),
		hit_in3				=> std_logic_vector(to_unsigned(4*i+2,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+3)-1 downto UNPACKER_HITSIZE*(4*i+2)),
		hit_ena3				=> sorterhits_ena(4*i+2),
		hit_in4				=> std_logic_vector(to_unsigned(4*i+3,4)) & sorterhits(UNPACKER_HITSIZE*(4*i+4)-1 downto UNPACKER_HITSIZE*(4*i+3)),
		hit_ena4				=> sorterhits_ena(4*i+3),
		time_in1				=> std_logic_vector(to_unsigned(4*i  ,4)) & sortercoarsecounters(COARSECOUNTERSIZE*4*i+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*4*i),
		time_ena1			=> sortercoarsecounters_ena(4*i),
		time_in2				=> std_logic_vector(to_unsigned(4*i+1,4)) & sortercoarsecounters(COARSECOUNTERSIZE*(4*i+1)+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*(4*i+1)),
		time_ena2			=> sortercoarsecounters_ena(4*i+1),
		time_in3				=> std_logic_vector(to_unsigned(4*i+2,4)) & sortercoarsecounters(COARSECOUNTERSIZE*(4*i+2)+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*(4*i+2)),
		time_ena3			=> sortercoarsecounters_ena(4*i+2),
		time_in4				=> std_logic_vector(to_unsigned(4*i+3,4)) & sortercoarsecounters(COARSECOUNTERSIZE*(4*i+3)+BINCOUNTERSIZE-1 downto COARSECOUNTERSIZE*(4*i+3)),
		time_ena4			=> sortercoarsecounters_ena(4*i+3),
		hit_out				=> hit_ser_out(i),
		hit_ena				=> hit_ser_ena(i),
		time_out				=> time_ser_out(i),
		time_ena				=> time_ser_ena(i)
	);
end generate;

-- moved into partition
--process(clk125)
--begin
--	if(rising_edge(clk125)) then
--		hits_en				<= hit_ser_ena and (not time_ser_ena);
--		hit_ser_out_reg	<= hit_ser_out;
--	end if;
--end process;

-- count number of hits before sorter
gen_pre_sort_cnt:
for i in 0 to NSORTERINPUTS-1 generate
	pre_sort_cnt: hit_counter_simple 
	port map( 
		clock 		=> clk125,	-- should be the same as readclk of hitsorter
		reset_n 		=> resets_n(RESET_RO_BIT),
		coarse_ena	=> time_ser_ena(i),
		hits_ena_in	=> hit_ser_ena(i),
		counter		=> presort_per_serializer(i)
	);
end generate;

	
--process(clk125, resets_n(RESET_RO_BIT))
--begin
--	if(resets_n(RESET_RO_BIT) = '0') then
--		presort_sum_r_0	<= (others => '0');
----		presort_sum_r_1	<= (others => '0');
--		presort_sum			<= (others => '0');
--	elsif(rising_edge(clk125)) then
--	presort_sum_r_0	<= presort_per_serializer(0);
----		presort_sum_r_0	<= presort_per_serializer(0) + presort_per_serializer(1);
----		presort_sum_r_1	<= presort_per_serializer(2) + presort_per_serializer(3);
--		presort_sum			<= presort_sum_r_0;
----		presort_sum			<= presort_sum_r_0+presort_sum_r_1;
--	end if;
--end process;

readregs_slow(RECEIVER_PRESORT_TOP_REGISTER_R)(15 downto 0)		<= presort_per_serializer(0)(47 downto 32);--presort_sum(47 downto 32);
readregs_slow(RECEIVER_PRESORT_BOTTOM_REGISTER_R)	<= presort_per_serializer(0)(REG64_BOTTOM_RANGE);--presort_sum(REG64_BOTTOM_RANGE);
		
gen_hitsorter_bool:
if use_Hitsorter generate		
		
	sort: hitsorter 
	--	generic map(
	----		NINPUT	=> 1,
	--		NINPUT2	=> 2,
	--		NINPUT4	=> 1,
	--		NINPUT8	=> 1,
	--		NINPUT16	=> 1,
	--		HITSIZE	=> 2+2+8+8+6+10	-- serializer + link + Row + Col + Charge + TS	
	--	)
		port map(
			reset_n		=> resets_n(RESET_RO_BIT),
			writeclk		=> clk125,
			tsdivider	=> writeregs_reg(HITSORTER_TIMING_REGISTER_W)(HITSORTER_TSDIVIDER_RANGE),	--"0000",				-- TODO: to be set via register
			readclk		=> clk125,		
--			hits			=> hit_ser_out_reg,
--			hits_en		=> hits_en,
			hits_ser			=> hit_ser_out,
			hits_ser_en		=> hit_ser_ena,
			ts_en			=> time_ser_ena,
			tsdelay		=> writeregs_reg(HITSORTER_TIMING_REGISTER_W)(HITSORTER_TSDELAY_RANGE),	-- "1010000000",	-- "0000000000"	-- TODO: to be set via register
			fromtrigfifo	=> fromtrigfifo,
			trigfifoempty	=> telescope_trigfifoempty,
			fromhbfifo	=> fromhitbusfifo,
			hbfifoempty	=> telescope_hitbusfifoempty,
			data_out		=> telescope2mem,					
			out_ena		=> telescope2memena,
			out_eoe		=> telescope2memeoe,
			out_hit		=> sorter_out_hit,
			readtrigfifo	=> telescope_readtrigfifo,
			readhbfifo		=> telescope_readhitbusfifo,
			received_hits		=> readregs_slow(HITSORTER_RECEIVED_HITS_REGISTER_R),
			outoftime_hits		=> readregs_slow(HITSORTER_OUTOFTIME_HITS_REGISTER_R),
			intime_hits			=> readregs_slow(HITSORTER_INTIME_HITS_REGISTER_R),
			memwrite_hits		=> readregs_slow(HITSORTER_MEMWRITE_HITS_REGISTER_R),
			overflow_hits		=> readregs_slow(HITSORTER_OVERFLOW_HITS_REGISTER_R),
			sent_hits			=> readregs_slow(HITSORTER_SENT_HITS_REGISTER_R),
			break_counter		=> readregs_slow(HITSORTER_BREAKCOUNTER_REGISTER_R)
		);
		
end generate gen_hitsorter_bool;
	
-- count number of hits after sorter
post_sort_cnt: hit_counter_simple 
	port map( 
		clock 		=> clk125,	-- should be the same as readclk of hitsorter
		reset_n 		=> resets_n(RESET_RO_BIT),
		coarse_ena	=> '0',
		hits_ena_in	=> sorter_out_hit,
		counter		=> postsort_sum	-- TODO: write postsort_sum to a register
	);
	
readregs_slow(RECEIVER_POSTSORT_TOP_REGISTER_R)(15 downto 0)		<= postsort_sum(47 downto 32);
readregs_slow(RECEIVER_POSTSORT_BOTTOM_REGISTER_R)	<= postsort_sum(REG64_BOTTOM_RANGE);

	
--			
-- commented out telescope RO as it has to be adapted to #16 channels and 40b hit size first
--
--telescopero: telescope_ro 
--	port map(
--		clk				=> clk125,
--		readclk			=> clk125,
--		reset_n			=> resets_n(RESET_RO_BIT),									
--		hits_in			=> telescope_hits,			
--		hitena_in		=> telescope_hits_ena,	
--		counter_in		=> coarsecounters_bin,			
--		counterena_in	=> coarsecounters_bin_ena,	
--		timestamp		=> counter125,
--		hitsorter_cntdown				=> writeregs_reg(HITSORTER_CNTDOWN_REGISTER_W)(HITSORTER_CNTDOWN_RANGE),
--		hitsorter_cntdown_written	=> regwritten_reg(HITSORTER_CNTDOWN_REGISTER_W),
--		eventtime_sel	=> writeregs(HITSORTER_EVENTTIME_SEL_REG_W)(HITSORTER_COARSETIME_SEL_BIT),
--		hitmarker			=> writeregs_reg(RO_CHIPMARKER_REGISTER_W)(RO_HITMARKER_RANGE),
--		hitmarker_written => regwritten_reg(RO_CHIPMARKER_REGISTER_W),
--		tomem				=> telescope2mem,
--		tomem_ena		=> telescope2memena,
--		tomem_eoe		=> telescope2memeoe,
--		
----		errcounter_readwrite			=> errcounter_readwrite,
----		errcounter_writedone			=> errcounter_writedone,	
----		errcounter_zeroaddr			=> errcounter_zeroaddr,	
----		errcounter_hcnt_empty		=> errcounter_hcnt_empty,
----		errcounter_reset				=> errcounter_reset,
----		errcounter_read_old			=> errcounter_read_old,
----		errcounter_ignoredhits		=> errcounter_ignoredhits,
----		errcounter_ignoredblocks	=> errcounter_ignoredblocks,		
----		errcounter_identhit_read	=> errcounter_identhit_read,
----		errout_identhit_read			=> errout_identhit_read,
----		errcounter_identhit_write	=> errcounter_identhit_write,
----		errout_identhit_write		=> errout_identhit_write,
------		errcounter_readwrite		=> open,	--readregs_slow(HITSORTER_ERRCOUNT_READWRITE_REGISTER_R),
------		errcounter_writedone		=> readregs_slow(HITSORTER_ERRCOUNT_WRITEDONE_REGISTER_R),	
------		errcounter_zeroaddr		=> readregs_slow(HITSORTER_ERRCOUNT_ZEROADDR_REGISTER_R),	
------		errcounter_hcnt_empty	=> readregs_slow(HITSORTER_ERRCOUNT_HCNTEMPTY_REGISTER_R),
------		errcounter_reset			=> open,	--readregs_slow(HITSORTER_ERRCOUNT_RESET_REGISTER_R),
------		errcounter_read_old		=> open,	--readregs_slow(HITSORTER_ERRCOUNT_READOLD_REGISTER_R),
------		errcounter_ignoredhits	=> readregs_slow(HITSORTER_ERRCOUNT_IGNOREDHITS_REGISTER_R),
------		errcounter_ignoredblocks	=> open,	--readregs_slow(HITSORTER_ERRCOUNT_IGNOREDBLOCKS_REGISTER_R),
------		errcounter_identhit_read	=> readregs_slow(HITSORTER_ERRCOUNT_IGNOREDBLOCKS_REGISTER_R),	--1E
------		errout_identhit_read			=> readregs_slow(HITSORTER_ERRCOUNT_RESET_REGISTER_R),	--1F
------		errcounter_identhit_write	=> readregs_slow(HITSORTER_ERRCOUNT_READOLD_REGISTER_R),	--2F
------		errout_identhit_write		=> readregs_slow(HITSORTER_ERRCOUNT_READWRITE_REGISTER_R),	--30
--
----		readtrigfifo				=> telescope_readtrigfifo,
----		fromtrigfifo				=> fromtrigfifo,
----		trigfifoempty				=> telescope_trigfifoempty,
----		presort_hits				=> presort_hits,
----		presort_sum					=> presort_sum,			
----		postsort_hits				=> postsort_hits,	
----		postsort_sum				=> postsort_sum,
--
--		binhitout					=> muxbinhit,
--		binhitenaout				=> muxbinhitena,
--		bincounterenaout 			=> muxcounterena--,
----		reset_readinithitsortmem	=> reset_readinithitsortmem_n,
----		reset_writeinithitsortmem	=> reset_writeinithitsortmem_n, 
----		cnt_hitswritten			=> cnt_hitswritten,
----		cnt_hitsread				=> cnt_hitsread,
----		cnt_overflow				=> cnt_overflow,
----		cnt_coarse					=> cnt_coarse,
----		
----		readhbfifo					=> telescope_readhitbusfifo,
----		fromhbfifo					=> fromhitbusfifo,
----		hbfifoempty					=> telescope_hitbusfifoempty	
--		);
--		
--	hitsorter_mux	:	hitsorter_debug_mux
--	port map(
--		clk					=> clk125,
--		dataA0x				=> errcounter_readwrite,
--		dataA1x				=> errcounter_writedone,		
--		dataA2x				=> errcounter_zeroaddr,
--		dataA3x				=> errcounter_hcnt_empty,
--		dataA4x				=> errcounter_reset,
--		dataA5x				=> errcounter_read_old,
--		dataA6x				=> errcounter_ignoredhits,
--		dataA7x				=> errcounter_ignoredblocks,
--
--		dataB0x				=> errcounter_identhit_read,
--		dataB1x				=> errout_identhit_read,
--		dataB2x				=> errcounter_identhit_write,
--		dataB3x				=> errout_identhit_write,
--		dataB4x				=> x"00000000",
--		dataB5x				=> x"00000000",
--		dataB6x				=> x"00000000",			
--		dataB7x				=> x"00000000",	
--		
--		dataC0x				=> err_identhit_rx(REG64_TOP_RANGE),
--		dataC1x				=> err_identhit_rx(REG64_BOTTOM_RANGE),
--		dataC2x				=> err_identhit_sync(REG64_TOP_RANGE),
--		dataC3x				=> err_identhit_sync(REG64_BOTTOM_RANGE),
--		dataC4x				=> err_identhit_unpack(REG64_TOP_RANGE),
--		dataC5x				=> err_identhit_unpack(REG64_BOTTOM_RANGE),
--		dataC6x				=> err_identhit_degray(REG64_TOP_RANGE),
--		dataC7x				=> err_identhit_degray(REG64_BOTTOM_RANGE),
--		
--		dataD0x				=> cnt_hitswritten(REG64_TOP_RANGE),
--		dataD1x				=> cnt_hitswritten(REG64_BOTTOM_RANGE),
--		dataD2x				=> cnt_hitsread(REG64_TOP_RANGE),
--		dataD3x				=> cnt_hitsread(REG64_BOTTOM_RANGE),
--		dataD4x				=> cnt_overflow(REG64_TOP_RANGE),
--		dataD5x				=> cnt_overflow(REG64_BOTTOM_RANGE),
--		dataD6x				=> cnt_coarse(REG64_TOP_RANGE),			
--		dataD7x				=> cnt_coarse(REG64_BOTTOM_RANGE),	
--
--		sel 					=> writeregs_reg(DEBUG_CHIP_SELECT_REGISTER_W)(HITSORTER_DEBUG_SELECT_RANGE),
--		dout0					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R),
--		dout1					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+1),
--		dout2					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+2),
--		dout3					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+3),
--		dout4					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+4),
--		dout5					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+5),
--		dout6					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+6),
--		dout7					=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R+7)
--		);
		
--	Hitsorter_mux : histo_mux
--	PORT map(
--		clock			=> clk125,
--		data0x		=> errcounter_readwrite,
--		data1x		=> errcounter_writedone,		
--		data2x		=> errcounter_zeroaddr,
--		data3x		=> errcounter_hcnt_empty,
--		data4x		=> errcounter_reset,
--		data5x		=> errcounter_read_old,
--		data6x		=> errcounter_ignoredhits,
--		data7x		=> errcounter_ignoredblocks,
--		data8x		=> errcounter_identhit_read,
--		data9x		=> errout_identhit_read,
--		data10x		=> errcounter_identhit_write,
--		data11x		=> errout_identhit_write,
--		data12x		=> x"00000000",
--		data13x		=> x"00000000",
--		data14x		=> x"00000000",			
--		data15x		=> x"00000000",	
--		data16x		=> x"00000000",
--		data17x		=> x"00000000",
--		data18x		=> x"00000000",
--		data19x		=> x"00000000",
--		data20x		=> cnt_hitswritten(REG64_TOP_RANGE),
--		data21x		=> cnt_hitswritten(REG64_BOTTOM_RANGE),
--		data22x		=> cnt_hitsread(REG64_TOP_RANGE),
--		data23x		=> cnt_hitsread(REG64_BOTTOM_RANGE),
--		data24x		=> cnt_overflow(REG64_TOP_RANGE),
--		data25x		=> cnt_overflow(REG64_BOTTOM_RANGE),
--		data26x		=> cnt_coarse(REG64_TOP_RANGE),			
--		data27x		=> cnt_coarse(REG64_BOTTOM_RANGE),	
--		data28x		=> x"00000000",
--		data29x		=> x"00000000",		
--		data30x		=> err_identhit_rx(REG64_TOP_RANGE),
--		data31x		=> err_identhit_rx(REG64_BOTTOM_RANGE),
--		data32x		=> err_identhit_sync(REG64_TOP_RANGE),
--		data33x		=> err_identhit_sync(REG64_BOTTOM_RANGE),
--		data34x		=> err_identhit_unpack(REG64_TOP_RANGE),
--		data35x		=> err_identhit_unpack(REG64_BOTTOM_RANGE),
--		data36x		=> err_identhit_degray(REG64_TOP_RANGE),
--		data37x		=> err_identhit_degray(REG64_BOTTOM_RANGE),
--		data38x		=> x"00000000",
--		data39x		=> x"00000000",	
--		sel			=> writeregs_reg(DEBUG_CHIP_SELECT_REGISTER_W)(HITSORTER_DEBUG_SELECT_RANGE),
--		result		=> readregs_slow(HITSORTER_ERRCOUNT_DEBUG_REGISTER_R)
--	);		

-- adapt for all links!		
--presort_mux_LSB : memorymux
--PORT MAP
--	(
--		clock			=> clk125,
--		data0x		=> presort_sum(REG64_BOTTOM_RANGE),
--		data1x		=> presort_hits(0)(REG64_BOTTOM_RANGE),		
--		data2x		=> presort_hits(1)(REG64_BOTTOM_RANGE),
--		data3x		=> presort_hits(2)(REG64_BOTTOM_RANGE),
--		data4x		=> presort_hits(3)(REG64_BOTTOM_RANGE),
--		data5x		=> x"00000000",
--		data6x		=> x"00000000",
--		data7x		=> x"00000000",
--		sel			=> sort_mux_sel,
--		result		=> readregs_slow(RECEIVER_PRESORT_BOTTOM_REGISTER_R)
--	);				
	
--presort_mux_MSB : memorymux
--PORT MAP
--	(
--		clock			=> clk125,
--		data0x		=> presort_sum(REG64_TOP_RANGE),
--		data1x		=> presort_hits(0)(REG64_TOP_RANGE),		
--		data2x		=> presort_hits(1)(REG64_TOP_RANGE),
--		data3x		=> presort_hits(2)(REG64_TOP_RANGE),
--		data4x		=> presort_hits(3)(REG64_TOP_RANGE),
--		data5x		=> x"00000000",
--		data6x		=> x"00000000",
--		data7x		=> x"00000000",
--		sel			=> sort_mux_sel,
--		result		=> readregs_slow(RECEIVER_PRESORT_TOP_REGISTER_R)
--	);				
	
--
--postsort_mux_LSB : memorymux
--PORT MAP
--	(
--		clock			=> clk125,
--		data0x		=> postsort_sum(REG64_BOTTOM_RANGE),		
--		data1x		=> postsort_hits(0)(REG64_BOTTOM_RANGE),
--		data2x		=> postsort_hits(1)(REG64_BOTTOM_RANGE),
--		data3x		=> postsort_hits(2)(REG64_BOTTOM_RANGE),
--		data4x		=> postsort_hits(3)(REG64_BOTTOM_RANGE),
--		data5x		=> x"00000000",
--		data6x		=> x"00000000",
--		data7x		=> x"00000000",
--		sel			=> sort_mux_sel,
--		result		=> readregs_slow(RECEIVER_POSTSORT_BOTTOM_REGISTER_R)
--	);				
--	
--postort_mux_MSB : memorymux
--PORT MAP
--	(
--		clock			=> clk125,
--		data0x		=> postsort_sum(REG64_TOP_RANGE),			
--		data1x		=> postsort_hits(0)(REG64_TOP_RANGE),
--		data2x		=> postsort_hits(1)(REG64_TOP_RANGE),
--		data3x		=> postsort_hits(2)(REG64_TOP_RANGE),
--		data4x		=> postsort_hits(3)(REG64_TOP_RANGE),
--		data5x		=> x"00000000",
--		data6x		=> x"00000000",
--		data7x		=> x"00000000",
--		sel			=> sort_mux_sel,
--		result		=> readregs_slow(RECEIVER_POSTSORT_TOP_REGISTER_R)
--	);		
					


------------------------------------------------------------------------------------		
-- I guess the code below does not require any changes, unless new readout modes are implemented
--////////////////////////TRIGGER LOGiC
telescope_trigfifoempty			<= trigfifoempty  when romode=3 else '1';
allsinglezs2_trigfifoempty		<= trigfifoempty  when romode=4 else '1';

readtrigfifo	<= muxreadtrigfifo0	when romode=0 else
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
--////////////////////////TRIGGER LOGIC done

--////////////////////////HITBUS LOGIC
-- only selected RO mode sees if there are triggers, the other modes just skip
telescope_hitbusfifoempty			<= hitbusfifoempty  when romode=3 else '1';
allsinglezs2_hitbusfifoempty		<= hitbusfifoempty  when romode=4 else '1';

-- only selected readout mode may read from the FIFO
readhitbusfifo	<= muxreadhitbusfifo0	when romode=0 else
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
--////////////////////////HITBUS LOGIC done	



--err_identhit_rx		<= errcounter_identhit_rx		& errout_identhit_rx;
--err_identhit_sync		<= errcounter_identhit_sync	& errout_identhit_sync;
--err_identhit_unpack	<= errcounter_identhit_unpack	& errout_identhit_unpack;
--err_identhit_degray	<= errcounter_identhit_degray	& errout_identhit_degray;

------------------------  Multiplex output and drive memory ------------------------

-- Readout modes
romode		<= writeregs_reg(RO_MODE_REGISTER_W)(RO_MODE_RANGE);
-- 0: Raw data from links (k signal is lost) / 4 * 8 bits
with mux_gx_register select muxdata0 <=
		rx_data_in(31 downto 0) 	when "00",
		rx_data_in(63 downto 32) 	when "01",
		rx_data_in(95 downto 64)	when "10",
		rx_data_in(127 downto 96)	when "11";	
muxena0		<= '1';	-- when rx_syncstatus(0) = '1' or rx_syncstatus(1) = '1' or rx_syncstatus(2) = '1' or rx_syncstatus(3) = '1'
							-- else '0';
muxeoe0		<= '0';					

-- 1: Single chip (A front) readout
muxdata1		<= single2mem;
muxena1		<= single2memena;
muxeoe1		<= single2memeoe;

-- 2: Single chip (A front) readout, zero suppressed
muxdata2		<= singlezs2mem;
muxena2		<= singlezs2memena;
muxeoe2		<= singlezs2memeoe;

-- 3: Telescope readout
muxdata3		<= telescope2mem;
muxena3		<= telescope2memena;
muxeoe3		<= telescope2memeoe;


-- 4: Multiplexed Hitblocks, using single readout of all chips, zero suppressed
muxdata4		<= allsinglezs2mem;
muxena4		<= allsinglezs2memena;
muxeoe4		<= allsinglezs2memeoe;

-- 5: Not used
muxdata5		<= (others => '0');
muxena5		<= '0';
muxeoe5		<= '0';
-- 6: Not used
muxdata6		<= (others => '0');
muxena6		<= '0';
muxeoe6		<= '0';
-- 7: Not used
muxdata7		<= (others => '0');
muxena7		<= '0';
muxeoe7		<= '0';

memmux: memorymux 
	PORT MAP
	(
		clock			=> clk125,
		data0x		=> muxdata0,
		data1x		=> muxdata1,
		data2x		=> muxdata2,
		data3x		=> muxdata3,
		data4x		=> muxdata4,
		data5x		=> muxdata5,
		data6x		=> muxdata6,
		data7x		=> muxdata7,
		sel			=> romode,
		result		=> muxtomem
	);

enamux: enablemux 
	PORT MAP
	(
		clock			=> clk125,
		data0			=> muxena0,
		data1			=> muxena1,
		data2			=> muxena2,
		data3			=> muxena3,
		data4			=> muxena4,
		data5			=> muxena5,
		data6			=> muxena6,
		data7			=> muxena7,
		sel			=> romode,
		result		=> muxenatomem
	);

eoemux: enablemux 
	PORT MAP
	(
		clock			=> clk125,
		data0			=> muxeoe0,
		data1			=> muxeoe1,
		data2			=> muxeoe2,
		data3			=> muxeoe3,
		data4			=> muxeoe4,
		data5			=> muxeoe5,
		data6			=> muxeoe6,
		data7			=> muxeoe7,
		sel			=> romode,
		result		=> muxeoetomem
	);	
	
-- Write to memory
readmem_addr		<= readmem_addr_reg;
readmem_clk			<= clk125;

readregs_slow(MEM_ADDRESS_REGISTER_R)(WRITE_ADDRESS_RANGE)	<= readmem_addr_reg;
readregs_slow(MEM_ADDRESS_REGISTER_R)(EOE_ADDRESS_RANGE)		<= readmem_eoeaddr_reg;

process(resets_n(RESET_ROMEMWRITER_BIT),clk125)
begin
if(resets_n(RESET_ROMEMWRITER_BIT) = '0') then
	readmem_addr_reg		<= (others => '0');
	readmem_wren			<= '0';
	readmem_eoe				<= '0';
	readmem_eoeaddr_reg	<= (others => '0');
elsif(clk125'event and clk125 = '1') then
	if(writeregs_reg(RO_MODE_REGISTER_W)(RO_ENABLE_BIT) = '1') then
		readmem_data			<= muxtomem;
		readmem_wren			<= muxenatomem;
		readmem_eoe				<= muxeoetomem;
		if(muxenatomem = '1') then
			readmem_addr_reg	<= readmem_addr_reg + '1';
			if(muxeoetomem = '1')then
				readmem_eoeaddr_reg	<= readmem_addr_reg + '1';
			end if;			
		end if;
	else
		readmem_wren			<= '0';
		readmem_eoe				<= '0';
	end if;
end if;
end process;


end rtl;
