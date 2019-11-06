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

entity mupix_datapath is
generic(
	NCHIPS: 				integer :=  4;
	NLVDS: 				integer := 16;
	NSORTERINPUTS: 	integer :=  4	--up to 4 LVDS links merge to one sorter
);
port (
	i_reset_n:			in std_logic;
	i_reset_n_lvds:	in std_logic;
	
	i_clk:				in std_logic;
	i_clk125:			in std_logic;
	
	lvds_data_in:		in std_logic_vector(NLVDS-1 downto 0);
	
	write_sc_regs:		in reg32array_128;
	read_sc_regs: 		out reg32array_128;

	o_fifo_rdata    : out   std_logic_vector(35 downto 0);
	o_fifo_rempty   : out   std_logic;
	i_fifo_rack     : in    std_logic--;
);
end mupix_datapath;

architecture rtl of mupix_datapath is

signal reset	: std_logic;
signal lvds_pll_locked		: std_logic_vector(1 downto 0);
signal lvds_runcounter		: links_reg32;
signal lvds_errcounter		: links_reg32;	

-- signals after mux
signal rx_data				: inbyte_array;
signal rx_k					: std_logic_vector(NLVDS-1 downto 0);
signal data_valid			: std_logic_vector(NLVDS-1 downto 0);

-- hits + flag to indicate a word as a hit, after unpacker
signal hits 				: std_logic_vector(NCHIPS*UNPACKER_HITSIZE-1 downto 0);
signal hits_ena			: std_logic_vector(NCHIPS-1 downto 0);

-- hits after gray-decoding
signal binhits 			: std_logic_vector(NCHIPS*UNPACKER_HITSIZE-1 downto 0);
signal binhits_ena		: std_logic_vector(NCHIPS-1 downto 0);

-- flag to indicate link, after unpacker
signal link_flag 		: std_logic_vector(NCHIPS-1 downto 0);

-- link flag is pipelined once because hits are gray decoded
signal link_flag_del 	: std_logic_vector(NCHIPS-1 downto 0);

-- counter + flag to indicate word as a counter, after unpacker
signal coarsecounters 		: std_logic_vector(NCHIPS*COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena 	: std_logic_vector(NCHIPS-1 downto 0);

-- counter is pipelined once because hits are gray decoded
signal coarsecounters_del 		: std_logic_vector(NCHIPS *COARSECOUNTERSIZE-1 downto 0);
signal coarsecounters_ena_del 	: std_logic_vector(NCHIPS-1 downto 0);

-- error signal output from unpacker
signal unpack_errorcounter	: links_reg32;

-- writeregisters are registered once to reduce long combinational paths
signal writeregs_reg				: reg32array;
--signal regwritten_reg				: std_logic_vector(NREGISTERS-1 downto 0); 
signal s_buf_data				: std_logic_vector(35 downto 0);
signal sync_fifo_empty		: std_logic;
signal s_buf_full				: std_logic;
signal s_buf_almost_full	: std_logic;


signal s_buf_data_125		: std_logic_vector(33 downto 0);
signal s_buf_wr_125			: std_logic;

signal counter125 			: std_logic_vector(63 downto 0);

signal rx_state				: std_logic_vector(NLVDS*2-1 downto 0);

signal MULTICHIP_RO_OVERFLOW : std_logic_vector(31 downto 0);

begin

	reset <= not i_reset_n;

	u_common_fifo : work.common_fifo
	port map (
		clock           => i_clk,
		sclr            => reset,
		data            => s_buf_data,
		wrreq           => not sync_fifo_empty,
		full            => s_buf_full,
		almost_full     => s_buf_almost_full,
		empty           => o_fifo_rempty,
		q               => o_fifo_rdata,
		rdreq           => i_fifo_rack--,
	);
	
	e_two_clk_sync_fifo : work.two_clk_sync_fifo
	port map
	(
		data			=> "00" & s_buf_data_125,
		rdclk			=> i_clk,
		aclr			=> reset,
		rdreq			=> not sync_fifo_empty,
		wrclk			=> i_clk125,
		wrreq			=> s_buf_wr_125,
		q				=> s_buf_data,
		rdempty		=> sync_fifo_empty,
		wrfull		=> open--,
	);

	writregs_clocking : process(i_clk)
	begin
		if(rising_edge(i_clk))then
			for I in 63 downto 0 loop
				writeregs_reg(I)	<= write_sc_regs(I);
			end loop;
		end if;
	end process writregs_clocking;
	
------------------------------------------------------------------------------------
---------------------- LVDS Receiver part ------------------------------------------
	lvds_block : work.receiver_block_mupix
	generic map(
		NINPUT	=> NLVDS,
		NCHIPS	=> NCHIPS
	)
	port map(
		reset_n				=> i_reset_n_lvds,
		checker_rst_n		=> i_reset_n,
		rx_in				=> lvds_data_in,
		rx_inclock_A		=> i_clk125,
		rx_inclock_B		=> i_clk125,

		rx_state			=> rx_state,
		rx_ready			=> data_valid,
		rx_data				=> rx_data,
		rx_k				=> rx_k,
		pll_locked			=> lvds_pll_locked,	-- write to some register!
	
		rx_runcounter		=> lvds_runcounter,	-- read_sc_regs
		rx_errorcounter		=> lvds_errcounter,	-- would be nice to add some error counter
		nios_clock			=> i_clk	
	);
	
	-- hard for now
	read_sc_regs(RX_STATE_RECEIVER_0) <= rx_state(31 downto 0);
	read_sc_regs(RX_STATE_RECEIVER_1) <= rx_state(63 downto 32);
	read_sc_regs(LVDS_PLL_LOCKED_REG)(1 downto 0) <= lvds_pll_locked;
	read_sc_regs(MULTICHIP_RO_OVERFLOW_REG)		  <= MULTICHIP_RO_OVERFLOW;
	GEN_LVDS_REGS:
	FOR I in 0 to NLVDS - 1 GENERATE
		read_sc_regs(LVDS_RUNCOUNTER_REG + I) <= lvds_runcounter(I);
		read_sc_regs(LVDS_ERRCOUNTER_REG + I) <= lvds_errcounter(I);
	END GENERATE GEN_LVDS_REGS;
	

--------------------------------------------------------------------------------------
--------------------- Unpack the data ------------------------------------------------	
	genunpack:
	FOR i in 0 to NCHIPS-1 GENERATE	
	-- we currently only use link 0 of each chip (up to 8 possible)
 
		unpacker_single : work.data_unpacker_new
		generic map(
			COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
			UNPACKER_HITSIZE	=> UNPACKER_HITSIZE
		)
		port map(
			reset_n				=> i_reset_n,
			clk					=> i_clk125,
			datain				=> rx_data(4*i), 
			kin					=> rx_k(4*i), 
			readyin				=> data_valid(4*i),
			is_atlaspix			=> '0',
			hit_out				=> hits((i+1)*UNPACKER_HITSIZE-1 downto i*UNPACKER_HITSIZE),
			hit_ena				=> hits_ena(i),
			coarsecounter		=> coarsecounters((i+1)*COARSECOUNTERSIZE-1 downto i*COARSECOUNTERSIZE),
			coarsecounter_ena	=> coarsecounters_ena(i),
			link_flag			=> link_flag(i),
			errorcounter		=> unpack_errorcounter(i)
		);	
		
		degray_single : work.hit_ts_conversion 
		port map(
			reset_n				=> i_reset_n,
			clk					=> i_clk125, 
			invert_TS			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_INVERT_BIT),
			invert_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_INVERT_BIT),
			gray_TS				=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS_GRAY_BIT),
			gray_TS2			=> writeregs_reg(TIMESTAMP_GRAY_INVERT_REGISTER_W)(TS2_GRAY_BIT),
			hit_in				=> hits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_in			=> hits_ena(i),
			hit_out				=> binhits(UNPACKER_HITSIZE*(i+1)-1 downto UNPACKER_HITSIZE*i),
			hit_ena_out			=> binhits_ena(i)
		);

	END GENERATE genunpack;
	
	-- delay cc by one cycle to be in line with hit
	-- Seb: new degray - back to one
	process(i_clk125)
	begin
		if(i_clk125'event and i_clk125 = '1') then
			coarsecounters_del			<= coarsecounters;
			coarsecounters_ena_del		<= coarsecounters_ena;
			link_flag_del				<= link_flag;
		end if;
	end process;	
	

-----------------------------------------------------------------------------------------------------		
-------------------------- For all chips: Single chip RO mode, zero suppressed ----------------------

	process(i_clk125, i_reset_n)
	begin
		if(i_reset_n = '0')then
			counter125	<= (others => '0');
		elsif(rising_edge(i_clk125))then
			counter125	<=  counter125 + 1;
		end if;
	end process;

	multirozs : work.multichip_ro_zerosupressed
	generic map(
		COARSECOUNTERSIZE	=> COARSECOUNTERSIZE,
		HITSIZE				=> UNPACKER_HITSIZE,
		NCHIPS 				=> NCHIPS
	)
	port map(
		clk						=> i_clk125,												
		reset_n					=> i_reset_n,		
		counter125				=> counter125,
		link_flag				=> link_flag_del,		
		hits_in					=> binhits,			
		hits_ena				=> binhits_ena,	
		coarsecounters			=> coarsecounters_del,
		coarsecounters_ena		=> coarsecounters_ena_del,
		prescale				=> writeregs_reg(RO_PRESCALER_REGISTER_W)(RO_PRESCALER_RANGE),
--		is_shared				=> writeregs_reg(LINK_REGISTER_W)(LINK_SHARED_RANGE),		
		tomemdata				=> s_buf_data_125,
		tomemena				=> s_buf_wr_125,
		tomemeoe				=> open,
		errcounter_overflow 	=> MULTICHIP_RO_OVERFLOW,
		errcounter_sel_in		=> writeregs_reg(DEBUG_CHIP_SELECT_REGISTER_W)(CHIPRANGE-1 downto 0)
	);	


end rtl;