	-- stic 3 data receiver
-- Simon Corrodi based on KIP DAQ
-- July 2017
-- Konrad Briggl updated using lvds deserializer instead of gbt, preparation for multiple channels
-- April 2019
-- May 2019: Added frame-collecting multiplexer, prbs decoder and common buffer (standard fifo)
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;

use work.daq_constants.all;
use work.mutrig_constants.all;

entity mutrig_datapath is
generic(
	N_ASICS : positive := 1;
	LVDS_PLL_FREQ : real := 125.0;
	LVDS_DATA_RATE : positive := 1250;
	GEN_DUMMIES : boolean := TRUE
);
port (
	i_rst			: in  std_logic;				-- logic reset
	i_stic_txd		: in  std_logic_vector( N_ASICS-1 downto 0);	-- serial data
	i_refclk_125		: in  std_logic;                 		-- ref clk for pll
	i_ts_clk		: in  std_logic;                 		-- ref clk for global timestamp
	i_ts_rst		: in  std_logic;				-- global timestamp reset, high active

	--interface to asic fifos
	i_clk_core		: in  std_logic; --fifo reading side clock
	o_fifo_empty		: out std_logic;
	o_fifo_data		: out std_logic_vector(35 downto 0);
	i_fifo_rd		: in  std_logic;
	--slow control
	i_SC_disable_dec	: in std_logic;
	i_SC_mask		: in std_logic_vector( N_ASICS-1 downto 0);
	i_SC_datagen_enable	: in std_logic;
	i_SC_datagen_shortmode	: in std_logic;
	i_SC_datagen_count	: in std_logic_vector(9 downto 0);
	--monitors
	o_receivers_usrclk	: out std_logic;              		-- pll output clock
	o_receivers_pll_lock	: out std_logic;			-- pll lock flag
	o_receivers_dpa_lock	: out std_logic_vector( N_ASICS-1 downto 0);			-- dpa lock flag per channel
	o_receivers_ready	: out std_logic_vector( N_ASICS-1 downto 0);-- receiver output ready flag
	o_frame_desync		: out std_logic;
	o_buffer_full		: out std_logic
);
end entity mutrig_datapath;


architecture RTL of mutrig_datapath is

component frame_rcv is
	generic (
		EVENT_DATA_WIDTH	: positive := 48;
		N_BYTES_PER_WORD	: positive := 6;
		N_BYTES_PER_WORD_SHORT	: positive := 3
	);
	port(
		i_rst              : in std_logic;
		i_clk              : in std_logic;
		i_data             : in std_logic_vector(7 downto 0);
		i_byteisk          : in std_logic;
		i_dser_no_sync     : in std_logic; -- deserialzer not synced after rst

		o_frame_number     : out std_logic_vector(15 downto 0);
		o_frame_info       : out std_logic_vector(15 downto 0);
		o_frame_info_ready : out std_logic;
		o_new_frame        : out std_logic;
		o_word             : out std_logic_vector(EVENT_DATA_WIDTH -1 downto 0);
		o_new_word         : out std_logic;

		o_end_of_frame     : out std_logic;
		o_crc_error        : out std_logic;
		o_crc_err_count    : out std_logic_vector(31 downto 0)
	);
end component;

component stic_dummy_data is
	port (
		i_clk            : in  std_logic;                                        -- byte clk
		i_reset          : in  std_logic;                                        -- async but at least one 125Mhz cycle long
		i_enable         : in  std_logic;                                        -- 
		i_fast           : in  std_logic;                                        -- if enable, fast data format
		i_cnt            : in  std_logic_vector( 9 downto 0);                    -- number of events per frame
		o_event_data     : out std_logic_vector(47 downto 0);   -- 
		o_event_ready    : out std_logic;                                        -- new word ready
		o_end_of_frame   : out std_logic;                                        -- end of frame: new_frame_info
		o_frame_number   : out std_logic_vector(15 downto 0);                    -- counter
		o_frame_info     : out std_logic_vector(15 downto 0);                    -- frame_flags(6) + frame_length(10)
		o_new_frame	   : out std_logic;                                        -- begin of new frame
		o_frame_info_rdy : out std_logic
	);
end component;

component mutrig_store is
port (
		i_clk_deser      : in  std_logic;
		i_clk_rd         : in  std_logic;					-- fast PCIe memory clk 
		i_reset          : in  std_logic;					-- reset, active low
		i_event_data     : in  std_logic_vector(47 downto 0);	-- event data from deserelizer
		i_event_ready    : in  std_logic;     
		i_new_frame      : in  std_logic;					-- start of frame
		i_frame_info_rdy : in  std_logic;					-- frame info ready (2 cycles after new_frame)
		i_end_of_frame   : in  std_logic;					-- end   of frame
		i_frame_info     : in  std_logic_vector(15 downto 0);
		i_frame_number   : in  std_logic_vector(15 downto 0);
		i_crc_error      : in  std_logic;
	--event data output inteface
		o_fifo_data	 : out std_logic_vector(55 downto 0);
		o_fifo_empty    :  out std_logic;
		i_fifo_rd	:  in std_logic;
	--monitoring, write-when-fill is prevented internally
		o_fifo_full       : out std_logic;					-- sync to i_clk_deser
		o_eventcounter   : out std_logic_vector(63 downto 0);			-- sync to i_clk_deser
		o_timecounter    : out std_logic_vector(63 downto 0);			-- sync to i_clk_deser
		i_SC_mask	: in std_logic						-- '1':  block any data from being written to the fifo
	);
end component; --mutrig_store;

component framebuilder_mux is
generic(
		N_INPUTS : integer;
		N_INPUTID_BITS : integer
);
port (
		i_coreclk        : in  std_logic;                                     -- system clock
		i_rst          : in  std_logic;                                     -- reset, active low

	--global timestamp
		i_timestamp_clk  : in  std_logic;	--125M timestamp clock
		i_timestamp_rst  : in  std_logic;	--timestamp reset, synced to i_timestamp_clk, high active

	--event data inputs interface
		i_source_data	 : in mutrig_evtdata_array_t(N_INPUTS-1 downto 0);
		i_source_empty   : in std_logic_vector(N_INPUTS-1 downto 0);
		o_source_rd   	 : out std_logic_vector(N_INPUTS-1 downto 0);

	--event data output interface to big buffer storage
		o_sink_data	 : out std_logic_vector(33 downto 0);		      -- event data output
		i_sink_full      :  in std_logic;
		o_sink_wr   	 : out std_logic;
	--monitoring, write-when-fill is prevented internally
		o_sync_error     : out std_logic;
		i_SC_mask	 : in std_logic_vector(N_INPUTS-1 downto 0)
);
end component; --framebuilder_mux;

component prbs_decoder is
port (
	--system
		i_coreclk	: in  std_logic;
		i_rst		: in  std_logic;
	--data stream input
		i_data	: in std_logic_vector(33 downto 0);
		i_valid	: in std_logic;
	--data stream output
		o_data	: out std_logic_vector(33 downto 0);
		o_valid	: out std_logic;
	--disable block (make transparent)
		i_SC_disable_dec : in std_logic
);
end component; --prbs_decoder;

component common_fifo
	PORT
	(
		clock		: IN STD_LOGIC ;
		data		: IN STD_LOGIC_VECTOR (35 DOWNTO 0);
		rdreq		: IN STD_LOGIC ;
		sclr		: IN STD_LOGIC ;
		wrreq		: IN STD_LOGIC ;
		almost_full		: OUT STD_LOGIC ;
		empty		: OUT STD_LOGIC ;
		full		: OUT STD_LOGIC ;
		q		: OUT STD_LOGIC_VECTOR (35 DOWNTO 0);
		usedw		: OUT STD_LOGIC_VECTOR (7 DOWNTO 0)
	);
end component;

subtype t_vector is std_logic_vector(N_ASICS-1 downto 0);
type t_array_64b is array (N_ASICS-1 downto 0) of std_logic_vector(64-1 downto 0);
type t_array_48b is array (N_ASICS-1 downto 0) of std_logic_vector(48-1 downto 0);
type t_array_16b is array (N_ASICS-1 downto 0) of std_logic_vector(16-1 downto 0);
type t_array_8b  is array (N_ASICS-1 downto 0) of std_logic_vector(8-1 downto 0);
type t_array_2b  is array (N_ASICS-1 downto 0) of std_logic_vector(2-1 downto 0);

  -- clocks

  -- serdes-frame_rcv
signal s_receivers_state	: std_logic_vector(2*N_ASICS-1 downto 0);
signal s_receivers_ready	: t_vector;
signal s_receivers_data		: std_logic_vector(8*N_ASICS-1 downto 0);
signal s_receivers_data_isk	: t_vector;

signal s_receivers_usrclk	: std_logic;
  -- frame_rcv/datagen - fifo: fifo side, frame-receiver side, dummy datagenerator side
signal s_crc_error,      s_rec_crc_error: t_vector;
signal s_frame_number,   s_rec_frame_number,   s_gen_frame_number   : t_array_16b;
signal s_frame_info,     s_rec_frame_info,     s_gen_frame_info     : t_array_16b;
signal s_new_frame,      s_rec_new_frame,      s_gen_new_frame      : t_vector;
signal s_frame_info_rdy, s_rec_frame_info_rdy, s_gen_frame_info_rdy : t_vector;
signal s_event_data,     s_rec_event_data,     s_gen_event_data     : t_array_48b;
signal s_event_ready,    s_rec_event_ready,    s_gen_event_ready    : t_vector;
signal s_end_of_frame,   s_rec_end_of_frame,   s_gen_end_of_frame   : t_vector;

--fifo - frame collector mux
signal s_fifos_empty 		: std_logic_vector(N_ASICS-1 downto 0);
signal s_fifos_data		: mutrig_evtdata_array_t(N_ASICS-1 downto 0);
signal s_fifos_rd		: std_logic_vector(N_ASICS-1 downto 0);
-- frame collector mux - prbs decoder 
signal s_buf_predec_data	: std_logic_vector(33 downto 0);
signal s_buf_predec_full 	: std_logic;
signal s_buf_predec_wr		: std_logic;
-- prbs decoder - mu3edataformat-writer - common fifo
signal s_buf_data		: std_logic_vector(33 downto 0);
signal s_buf_full 		: std_logic;	--internal only, almost_full is checked
signal s_buf_almost_full 	: std_logic;
signal s_buf_wr			: std_logic;

-- monitoring signals TODO: connect as needed
signal s_fifos_full      : t_vector;	--elastic fifo full flags
signal s_eventcounter   : t_array_64b;
signal s_timecounter    : t_array_64b;


begin
u_rxdeser: entity work.receiver_block
generic map(
	NINPUT => N_ASICS,
	LVDS_PLL_FREQ => LVDS_PLL_FREQ,
	LVDS_DATA_RATE => LVDS_DATA_RATE--,
)
port map(
	reset_n			=> not i_rst,
	reset_n_errcnt		=> '0',
	rx_in			=> i_stic_txd,
	rx_inclock		=> i_refclk_125,
	rx_state		=> s_receivers_state,
	rx_ready		=> s_receivers_ready,
	rx_data			=> s_receivers_data,
	rx_k			=> s_receivers_data_isk,
	rx_clkout		=> s_receivers_usrclk,
	pll_locked		=> o_receivers_pll_lock,
	rx_dpa_locked_out	=> o_receivers_dpa_lock,
	rx_runcounter		=> open,		--TODO: connect
	rx_errorcounter		=> open		--TODO: connect
);

o_receivers_ready <= s_receivers_ready;
o_receivers_usrclk <= s_receivers_usrclk;


gen_frame: for i in 0 to N_ASICS-1 generate begin
u_frame_rcv : frame_rcv
	generic map(
		EVENT_DATA_WIDTH	=> 48,
		N_BYTES_PER_WORD	=> 6,
		N_BYTES_PER_WORD_SHORT	=> 3
	)
	port map (
		i_rst			=> i_rst,
		i_clk			=> s_receivers_usrclk,
		i_data			=> s_receivers_data((i+1)*8-1 downto i*8),
		i_byteisk		=> s_receivers_data_isk(i),
		i_dser_no_sync		=> not s_receivers_ready(i),

		-- to mutrig-store instance
		o_frame_number		=> s_rec_frame_number(i),
		o_frame_info		=> s_rec_frame_info(i),
		o_frame_info_ready 	=> s_rec_frame_info_rdy(i),
		o_new_frame	 	=> s_rec_new_frame(i),
		o_word		 	=> s_rec_event_data(i),
		o_new_word	 	=> s_rec_event_ready(i),
		o_end_of_frame	 	=> s_rec_end_of_frame(i),

		o_crc_error	 	=> s_crc_error(i),
		o_crc_err_count		=> open
	);
	gen_dummy: if GEN_DUMMIES generate begin
		--data generator
		u_data_dummy : stic_dummy_data
			port map (
				i_reset			=> i_rst,
				i_clk			=> s_receivers_usrclk,
				--configuration
				i_enable		=> i_SC_datagen_enable,
				i_fast			=> i_SC_datagen_shortmode,
				i_cnt			=> i_SC_datagen_count,
				-- to mutrig-store instance
				o_frame_number		=> s_gen_frame_number(i),
				o_frame_info		=> s_gen_frame_info(i),
				o_frame_info_rdy 	=> s_gen_frame_info_rdy(i),
				o_new_frame	 	=> s_gen_new_frame(i),
				o_event_data	 	=> s_gen_event_data(i),
				o_event_ready	 	=> s_gen_event_ready(i),
				o_end_of_frame	 	=> s_gen_end_of_frame(i)
			);
	end generate;
end generate;

--multiplex between physical and generated data sent to the elastic buffers
gen_dummy: if GEN_DUMMIES generate begin
	s_frame_number		<= s_gen_frame_number	when i_SC_datagen_enable='1' else s_rec_frame_number;
	s_frame_info		<= s_gen_frame_info		when i_SC_datagen_enable='1' else s_rec_frame_info;
	s_frame_info_rdy 	<= s_gen_frame_info_rdy	when i_SC_datagen_enable='1' else s_rec_frame_info_rdy;
	s_new_frame	 	<= s_gen_new_frame		when i_SC_datagen_enable='1' else s_rec_new_frame;
	s_event_data	 	<= s_gen_event_data		when i_SC_datagen_enable='1' else s_rec_event_data;
	s_event_ready	 	<= s_gen_event_ready		when i_SC_datagen_enable='1' else s_rec_event_ready;
	s_end_of_frame	 	<= s_gen_end_of_frame	when i_SC_datagen_enable='1' else s_rec_end_of_frame;
end generate;
gen_dummy_not : if not GEN_DUMMIES generate begin
	s_frame_number		<= s_rec_frame_number;
	s_frame_info		<= s_rec_frame_info;
	s_frame_info_rdy 	<= s_rec_frame_info_rdy;
	s_new_frame	 	<= s_rec_new_frame;
	s_event_data	 	<= s_rec_event_data;
	s_event_ready	 	<= s_rec_event_ready;
	s_end_of_frame	 	<= s_rec_end_of_frame;
end generate;


rcv_fifo: for i in 0 to N_ASICS-1 generate begin
u_elastic_buffer : mutrig_store
port map(
	i_clk_deser      => s_receivers_usrclk,
	i_clk_rd         => i_clk_core,
	i_reset          => i_rst,
	i_event_data     => s_event_data(i),
	i_event_ready    => s_event_ready(i),
	i_new_frame      => s_new_frame(i),
	i_frame_info_rdy => s_frame_info_rdy(i),
	i_end_of_frame   => s_end_of_frame(i),
	i_frame_info     => s_frame_info(i),
	i_frame_number   => s_frame_number(i),
	i_crc_error      => s_crc_error(i),
--event data output inteface
	o_fifo_data	 => s_fifos_data(i),
	o_fifo_empty     => s_fifos_empty(i),
	i_fifo_rd	 => s_fifos_rd(i),
--monitoring, write-when-fill is prevented internally
	o_fifo_full      => s_fifos_full(i),
	o_eventcounter   => s_eventcounter(i),
	o_timecounter    => s_timecounter(i),
	i_SC_mask	 => i_SC_mask(i)
);
end generate;

--mux between asic channels
u_mux: framebuilder_mux
	generic map( 
		N_INPUTS => N_ASICS,
		N_INPUTID_BITS => 4 
	)
	port map(
		i_coreclk		=> i_clk_core,
		i_rst			=> i_rst,
		i_timestamp_clk		=> i_ts_clk,
		i_timestamp_rst		=> i_ts_rst,
	--event data inputs interface
    		i_source_data		=> s_fifos_data,
		i_source_empty		=> s_fifos_empty,
		o_source_rd		=> s_fifos_rd,
	--event data output interface to big buffer storage
		o_sink_data		=> s_buf_predec_data,
		i_sink_full		=> s_buf_predec_full,
		o_sink_wr		=> s_buf_predec_wr,
	--monitoring, errors, slow control
		o_sync_error		=> o_frame_desync,
		i_SC_mask		=> i_SC_mask
	);
--prbs decoder
s_buf_predec_full <= s_buf_almost_full;
u_decoder: prbs_decoder
	port map (
		i_coreclk	=> i_clk_core,
		i_rst		=> i_rst,
    		i_data		=> s_buf_predec_data,
    		i_valid		=> s_buf_predec_wr,
    		o_data		=> s_buf_data,
    		o_valid		=> s_buf_wr,
		i_SC_disable_dec=> i_SC_disable_dec
	);

    e_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 36--,
    )
    port map (
        clock           => i_clk_core,
        data            => "00" & s_buf_data,
        rdreq           => i_fifo_rd,
        sclr            => i_rst,
        wrreq           => s_buf_wr,
        almost_empty    => open,
        almost_full     => s_buf_almost_full,
        empty           => o_fifo_empty,
        full            => s_buf_full,
        q               => o_fifo_data,
        usedw           => open--,
    );

o_buffer_full <= s_buf_full;
end architecture RTL;
