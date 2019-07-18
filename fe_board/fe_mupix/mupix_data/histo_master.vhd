-----------------------------------------------------------------------------
-- MUPIX7 histogram master entity
--
-- June 2017
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-- was 17k ALUTs before!
-----------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.histo_components.all;
use work.hitsorter_components.all;
use work.mupix_constants.all;
use work.mupix_types.all;

entity histo_master is
generic(
	NCHIPS  : integer := 16;
	NHISTOS : integer := 64;
	COARSECOUNTERSIZE	: integer	:= 32;
	UNPACKER_HITSIZE	: integer	:= 40
);
	port (
		reset_n:					in		std_logic;
		clk:						in		std_logic;
-- ctrl		
		takedata:				in		std_logic;
		zeromem:					in		std_logic;
		written:					in		std_logic;
		readaddr:				in		std_logic_vector(15 downto 0);		
		sel:						in 	std_logic_vector(7 downto 0);	-- allows for 256 histograms
		hitmap_chipsel:		in 	std_logic_vector(1 downto 0);
-- raw data	in
		datain:					in		std_logic_vector(NCHIPS*8-1 downto 0);	
		kommain:					in 	std_logic_vector(NCHIPS-1 downto 0);
-- unpacker data in
		hit_in:					in 	std_logic_vector(NCHIPS*UNPACKER_HITSIZE-1 downto 0);
		hit_ena:					in 	std_logic_vector(NCHIPS-1 downto 0);
		coarsecounter_in:		in 	std_logic_vector(NCHIPS*COARSECOUNTERSIZE-1 downto 0);
		coarsecounter_ena:	in 	std_logic_vector(NCHIPS-1 downto 0);
		link_flag:				in 	std_logic_vector(NCHIPS-1 downto 0);
-- muxed hits in
		mux_hit_in:				in 	hit_array_t;
		mux_hit_ena:				in 	std_logic_vector(NSORTERINPUTS-1 downto 0);
		mux_coarsecounter_ena:	in 	std_logic_vector(NSORTERINPUTS-1 downto 0);
-- output
		dataout_MSB:			out	std_logic_vector(31 downto 0);
		dataout_LSB:			out	std_logic_vector(31 downto 0)			
		);		
end histo_master;
	

	
architecture RTL of histo_master is

type histo_array is array (NHISTOS-1 downto 0) of reg32;

signal histo_array_lsb : histo_array;
signal histo_array_msb : histo_array;

signal hitmap_sel_int 	: integer range 0 to 3;
signal hit_select			: std_logic_vector(4*UNPACKER_HITSIZE-1 downto 0);
signal hit_ena_select	: std_logic_vector(3 downto 0);
signal coarsecounter_ena_select	: std_logic_vector(3 downto 0);

begin

process(reset_n, clk)
begin
if(reset_n = '0')then
	dataout_MSB <= (others => '0');
	dataout_LSB <= (others => '0');
elsif(rising_edge(clk))then
	dataout_MSB <= histo_array_msb(conv_integer(sel));
	dataout_LSB <= histo_array_lsb(conv_integer(sel));
end if;
end process;

gen_komma_if:
if(NCHIPS>16) generate 

	i_kommas_histo_0: mupix7_histo_kommas 
	generic map(
		NCHIPS => NCHIPS/2
	) 
		port map(
			rstn		=> reset_n,
			clk		=> clk,
			takedata	=> takedata,
			zeromem	=> zeromem,
			datain	=> datain(8*NCHIPS/2-1 downto 0),
			kommain	=> kommain(NCHIPS/2-1 downto 0),
			readaddr	=> readaddr(5 downto 0),
			written	=> written,		
			dataout_msb	=> histo_array_msb(0),	
			dataout_lsb	=> histo_array_lsb(0)	
		);
		
	i_kommas_histo_1: mupix7_histo_kommas 
	generic map(
		NCHIPS => NCHIPS/2
	) 
		port map(
			rstn		=> reset_n,
			clk		=> clk,
			takedata	=> takedata,
			zeromem	=> zeromem,
			datain	=> datain(8*NCHIPS-1 downto 8*NCHIPS/2),
			kommain	=> kommain(NCHIPS-1 downto NCHIPS/2),
			readaddr	=> readaddr(5 downto 0),
			written	=> written,		
			dataout_msb	=> histo_array_msb(NSORTERINPUTS+2+3*NCHIPS),	
			dataout_lsb	=> histo_array_lsb(NSORTERINPUTS+2+3*NCHIPS)	
		);	
		
end generate gen_komma_if;

gen_komma:
if(NCHIPS<=16) generate
	i_kommas_histo: mupix7_histo_kommas 
	generic map(
		NCHIPS => NCHIPS
	) 
		port map(
			rstn		=> reset_n,
			clk		=> clk,
			takedata	=> takedata,
			zeromem	=> zeromem,
			datain	=> datain,
			kommain	=> kommain,
			readaddr	=> readaddr(5 downto 0),
			written	=> written,		
			dataout_msb	=> histo_array_msb(0),	
			dataout_lsb	=> histo_array_lsb(0)	
		);
end generate gen_komma;
	

gen_histo_cnt : 
for I in 0 to NCHIPS-1 generate 

--gen_only_dut:
--if(I=4) generate

	i_counter_histo : mupix7_histo_counter  
	port map(
		rstn			=> reset_n,
		clk			=> clk,
		takedata		=> takedata,
		zeromem		=> zeromem,
		coarsecounter_in	=> coarsecounter_in((I+1)*COARSECOUNTERSIZE-1 downto I*COARSECOUNTERSIZE),
		coarsecounter_ena	=> coarsecounter_ena(I),
		readaddr		=> readaddr(3 downto 0),
		written		=> written,		
		dataout_msb	=> histo_array_msb(I+1),	
		dataout_lsb	=> histo_array_lsb(I+1)	
	);
	
	i_histo_hits: mupix7_histo_hits 
	generic map(
		UNPACKER_HITSIZE => UNPACKER_HITSIZE
	) 
	port map(
		rstn			=> reset_n,
		clk			=> clk,
		takedata		=> takedata,
		zeromem		=> zeromem,
		written		=> written,				
		hit_in		=> hit_in((I+1)*UNPACKER_HITSIZE-1 downto I*UNPACKER_HITSIZE),
		hit_ena		=> hit_ena(I),
		coarsecounter_ena	=> coarsecounter_ena(I),	
		link_flag	=> link_flag(I),	
		
--		readaddr_row	=> readaddr(15 downto 8),
--		readaddr_col	=> readaddr(7 downto 0),		
		readaddr_ts		=> readaddr(9 downto 0),
		readaddr_multi	=> readaddr(5 downto 0),
		readaddr_chg	=> readaddr(5 downto 0),
		
--		dataout_matrix	=> open,--histo_array_msb(I+1+NCHIPS)(9 downto 0),
		dataout_ts		=> histo_array_lsb(I+1+NCHIPS),
		dataout_multi	=> histo_array_msb(I+1+2*NCHIPS),
		dataout_chg		=> histo_array_lsb(I+1+2*NCHIPS)
		);
		
--	histo_array_lsb(I+1+NLVDS)(31 downto 9) <= (others => '0');
		
--	i_histo_hit_ts_diff : mupix7_histo_hitTS_diff  
--		port map(
--		rstn			=> reset_n,
--		clk			=> clk,
--		takedata		=> takedata,
--		zeromem		=> zeromem,
--		binhittime	=> hit_in((I*UNPACKER_HITSIZE)+TIMESTAMPSIZE_MPX8-1 downto I*UNPACKER_HITSIZE),
--		binhittime_ena => hit_ena(I),
--		coarsecnt_ena 	=> coarsecounter_ena(I),
--		readaddr			=> readaddr(9 downto 0),
--		dataout_hitTS_diff	=>  histo_array_msb(I+1+NCHIPS)
--		);	
	
--	end generate gen_only_dut;
end generate gen_histo_cnt;

process(clk)
	begin
	if(rising_edge(clk))then
		hitmap_sel_int 				<= conv_integer(hitmap_chipsel);
		hit_select 						<= hit_in(4*(hitmap_sel_int+1)*UNPACKER_HITSIZE-1 downto 4*hitmap_sel_int*UNPACKER_HITSIZE);
		hit_ena_select					<= hit_ena(4*(hitmap_sel_int+1)-1 downto 4*hitmap_sel_int);
		coarsecounter_ena_select	<= coarsecounter_ena(4*(hitmap_sel_int+1)-1 downto 4*hitmap_sel_int);	
	end if;
end process;

--process(clk)
--	begin
--	if(rising_edge(clk))then
--		hitmap_sel_int 				<= 1;
--		hit_select 						<= hit_in(4*(hitmap_sel_int+1)*UNPACKER_HITSIZE-1 downto 4*hitmap_sel_int*UNPACKER_HITSIZE);
--		hit_ena_select					<= hit_ena(4*(hitmap_sel_int+1)-1 downto 4*hitmap_sel_int);
--		coarsecounter_ena_select	<= coarsecounter_ena(4*(hitmap_sel_int+1)-1 downto 4*hitmap_sel_int);
--	end if;
--end process;

--gen_mux_hitmap : 
--for I in 0 to NCHIPS/4-1 generate 

	i_mux_hitmap: mupix7_histo_hits_4links --not muxed
	generic map(
		UNPACKER_HITSIZE 	=> UNPACKER_HITSIZE,
		NCHIPS				=> 4		--links
	)
		port map(
		rstn			=> reset_n,
		clk			=> clk,
		takedata		=> takedata,
		zeromem		=> zeromem,
		written		=> written,					
		hit_in		=> hit_select,
		hit_ena		=> hit_ena_select,
		coarsecounter_ena	=> coarsecounter_ena_select,	
		
		readaddr_row	=> readaddr(15 downto 8),
		readaddr_col	=> readaddr(7 downto 0),		
			
		dataout_matrix	=> histo_array_msb(1+3*NCHIPS)(15 downto 0)
		);
--

--gen_muxed : 
--for I in 0 to NSORTERINPUTS-1 generate 
--	i_tsdiff_4links : mupix7_histo_hitTS_diff_4links 
--	port map(
--		rstn			=> reset_n,
--		clk			=> clk,
--		takedata		=> takedata,
--		zeromem		=> zeromem,
--		binhittime	=> mux_hit_in(I)(TIMESTAMPSIZE_MPX8-1 downto 0),
--		binhittime_ena => mux_hit_ena(I),
--		coarsecnt_ena 	=> mux_coarsecounter_ena(I),
--		readaddr			=> readaddr(9 downto 0),
--		dataout_hitTS_diff	=>  histo_array_msb(I+2+3*NCHIPS)
--		);
--end generate gen_muxed;


end RTL;
	