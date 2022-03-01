	-- stic 3 data receiver
-- Simon Corrodi based on KIP DAQ
-- July 2017
-- Konrad Briggl updated using lvds deserializer instead of gbt, preparation for multiple channels
-- April 2019
-- May 2019: Added frame-collecting multiplexer, prbs decoder and common buffer (standard fifo)
-- Oct 2019: Added generic to flip sign of input depending on PCB design
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;

use work.mutrig.all;


entity mutrig_datapath is
generic(
    IS_SCITILE : std_logic := '1';
    N_MODULES           : integer range 1 to 2 := 1;
    N_ASICS             : positive := 1;
    N_LINKS             : positive := 1;
    N_CC                : positive := 15; -- will be always 15
    LVDS_PLL_FREQ       : real := 125.0;
    LVDS_DATA_RATE      : real := 1250.0;
    GEN_DUMMIES         : boolean := TRUE;
    INPUT_SIGNFLIP : std_logic_vector(31 downto 0):=x"00000000";
    C_CHANNELNO_PREFIX_A: std_logic_vector:=""; --use prefix value as the first bits (MSBs) of the chip number field. Leave empty to append nothing and use all bits from Input # numbering
    C_CHANNELNO_PREFIX_B: std_logic_vector:=""
    --(e.g. Tiles,  one module with up to 16 ASICs, PREFIX="")
    --(e.g. Fibers, two modules with up to 4 ASICs each, PREFIX="00" ; "01" for A and B )
);
port (
    i_rst_core      : in  std_logic;    -- logic reset of digital core (buffer clear, 156MHz clock synced)
    i_rst_rx        : in  std_logic;    -- logic reset of lvds receivers (125MHz clock synced)
    i_stic_txd      : in  std_logic_vector(N_MODULES*N_ASICS-1 downto 0);   -- serial data
    i_refclk_125_A  : in  std_logic;    -- ref clk for lvds pll (A-Side) 
    i_refclk_125_B  : in  std_logic;    -- ref clk for lvds pll (B-Side)
    i_ts_clk        : in  std_logic;    -- ref clk for global timestamp
    i_ts_rst        : in  std_logic;    -- global timestamp reset, high active

    --interface to asic fifos
    i_clk_core                  : in  std_logic;    --fifo reading side clock
    o_fifo_data                 : out std_logic_vector(36*(N_LINKS-1)+35 downto 0);
    o_fifo_wr                   : out std_logic_vector(N_LINKS-1 downto 0);
    i_common_fifos_almost_full  : in  std_logic_vector(N_LINKS-1 downto 0); 

    --slow control
    i_SC_disable_dec            : in std_logic;
    i_SC_mask                   : in std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
	 i_SC_mask_rx                : in std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
    i_SC_datagen_enable         : in std_logic;
    i_SC_datagen_shortmode      : in std_logic;
    i_SC_datagen_count          : in std_logic_vector(9 downto 0);
    i_SC_rx_wait_for_all        : in std_logic;
    i_SC_rx_wait_for_all_sticky : in std_logic;
    
    --run control
    i_RC_may_generate           : in std_logic; --do not generate new frames for runstates that are not RUNNING, allows to let fifos run empty
    o_RC_all_done               : out std_logic; --all fifos empty, all data read

    -- lapse lapse counter
    i_en_lapse_counter          : in std_logic;
    i_upper_bnd                 : in std_logic_vector(N_CC - 1 downto 0);

    --monitors
    o_receivers_usrclk          : out std_logic;    -- pll output clock
    o_receivers_pll_lock        : out std_logic;    -- pll lock flag
    o_receivers_dpa_lock        : out std_logic_vector(N_MODULES*N_ASICS-1 downto 0);			-- dpa lock flag per channel
    o_receivers_ready           : out std_logic_vector(N_MODULES*N_ASICS-1 downto 0); -- receiver output ready flag
    o_frame_desync              : out std_logic_vector(1 downto 0);

    i_SC_reset_counters         : in std_logic; --synchronous to i_clk_core
    i_SC_counterselect          : in std_logic_vector(6 downto 0); --select counter to be read out. 2..0: counter type selection. 6..3: counter channel selection
    o_counter_numerator         : out std_logic_vector(31 downto 0); --gray encoded, different clock domains 
    o_counter_denominator_low   : out std_logic_vector(31 downto 0);
    o_counter_denominator_high  : out std_logic_vector(31 downto 0)--;
);

end entity mutrig_datapath;

architecture RTL of mutrig_datapath is

constant N_ASICS_TOTAL : natural :=N_MODULES*N_ASICS;

subtype t_vector is std_logic_vector(N_ASICS_TOTAL-1 downto 0);
type t_array_64b is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(64-1 downto 0);
type t_array_48b is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(48-1 downto 0);
subtype t_array_32b is work.util.slv32_array_t(N_ASICS_TOTAL-1 downto 0);
type t_array_16b is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(16-1 downto 0);
type t_array_8b  is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(8-1 downto 0);
type t_array_2b  is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(2-1 downto 0);

-- clocks

-- serdes-frame_rcv
signal s_receivers_state	: std_logic_vector(2*N_ASICS_TOTAL-1 downto 0);
signal s_receivers_ready	: t_vector;
signal s_receivers_data		: std_logic_vector(8*N_ASICS_TOTAL-1 downto 0);
signal s_receivers_data_isk	: t_vector;

signal s_receivers_usrclk	: std_logic;
signal s_receivers_all_ready    : std_logic;
signal s_receivers_block        : std_logic;

-- frame_rcv/datagen - fifo: fifo side, frame-receiver side, dummy datagenerator side
signal s_crc_error: t_vector;
signal s_frame_number,   s_rec_frame_number,   s_gen_frame_number   : t_array_16b;
signal s_frame_info,     s_rec_frame_info,     s_gen_frame_info     : t_array_16b;
signal s_new_frame,      s_rec_new_frame,      s_gen_new_frame      : t_vector;
signal s_frame_info_rdy, s_rec_frame_info_rdy, s_gen_frame_info_rdy : t_vector;
signal s_event_data,     s_rec_event_data,     s_gen_event_data     : t_array_48b;
signal s_event_ready,    s_rec_event_ready,    s_gen_event_ready    : t_vector;
signal s_end_of_frame,   s_rec_end_of_frame,   s_gen_end_of_frame   : t_vector;
signal s_frec_busy, s_gen_busy  : t_vector;
signal s_any_framegen_busy, s_any_framegen_busy_156 : std_logic;

--fifo - frame collector mux
signal s_fifos_empty 		: std_logic_vector(N_ASICS_TOTAL-1 downto 0):=(others =>'1');
signal s_fifos_data		: mutrig_evtdata_array_t(N_ASICS_TOTAL-1 downto 0);
signal s_fifos_rd		: std_logic_vector(N_ASICS_TOTAL-1 downto 0);

-- frame collector mux - prbs decoder 
signal s_A_mux_busy, s_B_mux_busy : std_logic :='0';
signal s_A_buf_predec_data	: std_logic_vector(33 downto 0);
signal s_A_buf_predec_full 	: std_logic;
signal s_A_buf_predec_wr		: std_logic;
signal s_B_buf_predec_data	: std_logic_vector(33 downto 0);
signal s_B_buf_predec_full 	: std_logic :='0';
signal s_B_buf_predec_wr		: std_logic;

-- prbs decoder - mu3edataformat-writer - common fifo
signal s_A_buf_data		: std_logic_vector(33 downto 0);
signal s_A_buf_data_reg		: std_logic_vector(33 downto 0);
signal s_A_buf_wr		: std_logic;
signal s_B_buf_data		: std_logic_vector(33 downto 0);
signal s_B_buf_data_reg		: std_logic_vector(33 downto 0);
signal s_B_buf_wr		: std_logic;

-- monitoring signals TODO: connect as needed
signal s_fifos_full           : t_vector;	--elastic fifo full flags
signal s_eventcounter         : t_array_32b;
signal s_timecounter          : t_array_64b;
signal s_crcerrorcounter      : t_array_32b;
signal s_framecounter         : t_array_64b;
signal s_prbs_wrd_cnt         : t_array_64b;
signal s_prbs_err_cnt         : t_array_32b;
signal s_receivers_runcounter : t_array_32b;
signal s_receivers_errorcounter : t_array_32b;
signal s_receivers_synclosscounter : t_array_32b;
signal s_SC_reset_counters_125_n : std_logic;

-- lapse counter signals
signal CC_corrected_A : std_logic_vector(N_CC - 1 downto 0);
signal CC_corrected_B : std_logic_vector(N_CC - 1 downto 0);

begin

rst_sync_counter : entity work.reset_sync
port map( i_reset_n => not i_SC_reset_counters, o_reset_n => s_SC_reset_counters_125_n, i_clk => s_receivers_usrclk);

u_rxdeser: entity work.receiver_block
generic map(
	IS_SCITILE 		=> IS_SCITILE,
	NINPUT 			=> N_ASICS_TOTAL,
	LVDS_PLL_FREQ 	=> LVDS_PLL_FREQ,
	LVDS_DATA_RATE => LVDS_DATA_RATE,
	INPUT_SIGNFLIP => INPUT_SIGNFLIP--,
)
port map(
	reset_n			=> not i_rst_rx,
	reset_n_errcnt	=> s_SC_reset_counters_125_n,
	rx_in				=> i_stic_txd,
	rx_inclock_A	=> i_refclk_125_A,
	rx_inclock_B	=> i_refclk_125_B,
	rx_state			=> s_receivers_state,
	rx_ready			=> s_receivers_ready,
	rx_data			=> s_receivers_data,
	rx_k				=> s_receivers_data_isk,
	rx_clkout		=> s_receivers_usrclk,
	pll_locked		=> o_receivers_pll_lock,
	rx_dpa_locked_out		=> o_receivers_dpa_lock,
	rx_runcounter			=> s_receivers_runcounter,
	rx_errorcounter		=> s_receivers_errorcounter,
	rx_synclosscounter	=> s_receivers_synclosscounter
);

o_receivers_ready <= s_receivers_ready;
o_receivers_usrclk <= s_receivers_usrclk;

--generate a pll-synchronous all-ready signal for the data receivers.
--this assures all start dumping data into the fifos at the same time, and we do not enter a deadlock scenario from the start
gen_ready_all : process (s_receivers_usrclk, i_rst_rx, s_receivers_ready, i_SC_mask_rx)
variable v_ready : std_logic_vector(N_ASICS_TOTAL-1 downto 0);
begin
if ( i_rst_rx='1' ) then
    s_receivers_all_ready<='0';
    --
elsif ( rising_edge(s_receivers_usrclk) ) then
    v_ready := s_receivers_ready or i_SC_mask_rx;
    if ( v_ready = ((v_ready'range)=>'1') ) then
        s_receivers_all_ready<='1';
    end if;
    --
end if;
end process;

--if i_SC_rx_wait_for_all is set, wait for all (not masked) receivers to become ready before letting any data pass through the frame unpacking blocks.
--if i_SC_rx_wait_for_all_sticky is set in addition, the all_ready property is sticky: once all receivers become ready do not block data again.
--            The sticky bit is cleared with i_reset
--            Otherwise, data is blocked as soon as one receiver is loosing the pattern or sync.
releasedata_p : process(s_receivers_usrclk, i_rst_rx)
begin
if ( rising_edge(s_receivers_usrclk) ) then
    if ( i_rst_rx = '1' ) then
        s_receivers_block <= '1';
        --
    elsif ( i_SC_rx_wait_for_all_sticky = '1' ) then
        if(s_receivers_all_ready='1') then
            s_receivers_block <= '0';
        end if;
    else
        s_receivers_block <= not s_receivers_all_ready;
    end if;
    --
end if;
end process;

gen_frame: for i in 0 to N_ASICS_TOTAL-1 generate begin
u_frame_rcv : entity work.frame_rcv
    generic map(
        EVENT_DATA_WIDTH	=> 48,
        N_BYTES_PER_WORD	=> 6,
        N_BYTES_PER_WORD_SHORT	=> 3
    )
    port map (
        i_rst           => i_rst_rx,
        i_clk           => s_receivers_usrclk,
        i_data          => s_receivers_data((i+1)*8-1 downto i*8),
        i_byteisk       => s_receivers_data_isk(i),
        i_enable        => i_RC_may_generate and not (s_receivers_block or (not s_receivers_ready(i) and not i_SC_rx_wait_for_all)),

        -- to mutrig-store instance
        o_frame_number      => s_rec_frame_number(i),
        o_frame_info        => s_rec_frame_info(i),
        o_frame_info_ready  => s_rec_frame_info_rdy(i),
        o_new_frame         => s_rec_new_frame(i),
        o_word              => s_rec_event_data(i),
        o_new_word          => s_rec_event_ready(i),
        o_end_of_frame      => s_rec_end_of_frame(i),
        o_busy              => s_frec_busy(i),

        o_crc_error         => s_crc_error(i),
        o_crc_err_count     => open
    );
    gen_dummy : if GEN_DUMMIES generate begin
        --data generator
        u_data_dummy : entity work.stic_dummy_data
            port map (
                i_reset         => i_rst_rx,
                i_clk           => s_receivers_usrclk,
                --configuration
                i_enable        => i_SC_datagen_enable and i_RC_may_generate,
                i_fast          => i_SC_datagen_shortmode,
                i_cnt           => i_SC_datagen_count,
                -- to mutrig-store instance
                o_frame_number      => s_gen_frame_number(i),
                o_frame_info        => s_gen_frame_info(i),
                o_frame_info_rdy    => s_gen_frame_info_rdy(i),
                o_new_frame         => s_gen_new_frame(i),
                o_event_data        => s_gen_event_data(i),
                o_event_ready       => s_gen_event_ready(i),
                o_end_of_frame      => s_gen_end_of_frame(i),
                o_busy              => s_gen_busy(i)
            );

        --multiplex between physical and generated data sent to the elastic buffers. Use busy from datagenerator to ensure safe takeover
        s_frame_number(i)       <= s_gen_frame_number(i)    when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_frame_number(i);
        s_frame_info(i)         <= s_gen_frame_info(i)      when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_frame_info(i);
        s_frame_info_rdy(i)     <= s_gen_frame_info_rdy(i)  when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_frame_info_rdy(i);
        s_new_frame(i)          <= s_gen_new_frame(i)       when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_new_frame(i);
        s_event_data(i)         <= s_gen_event_data(i)      when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_event_data(i);
        s_event_ready(i)        <= s_gen_event_ready(i)     when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_event_ready(i);
        s_end_of_frame(i)       <= s_gen_end_of_frame(i)    when (i_SC_datagen_enable='1' or s_gen_busy(i)='1') else s_rec_end_of_frame(i);
    end generate;

    gen_dummy_not : if not GEN_DUMMIES generate begin
        s_frame_number(i)   <= s_rec_frame_number(i);
        s_frame_info(i)     <= s_rec_frame_info(i);
        s_frame_info_rdy(i) <= s_rec_frame_info_rdy(i);
        s_new_frame(i)      <= s_rec_new_frame(i);
        s_event_data(i)     <= s_rec_event_data(i);
        s_event_ready(i)    <= s_rec_event_ready(i);
        s_end_of_frame(i)   <= s_rec_end_of_frame(i);
    end generate;

end generate;

p_frec_busy_sync: process (i_clk_core, s_receivers_usrclk)
begin
    if ( rising_edge(s_receivers_usrclk) ) then
        s_any_framegen_busy <= '0';
        if ( i_SC_datagen_enable = '1' and unsigned(s_gen_busy) /= 0 ) then
            s_any_framegen_busy <= '1';
        end if;
        if ( i_SC_datagen_enable='0' and unsigned(s_frec_busy) /= 0 ) then
            s_any_framegen_busy <= '1';
        end if;
    end if;
    if ( rising_edge(i_clk_core) ) then
        s_any_framegen_busy_156 <= s_any_framegen_busy;
    end if;
end process;

g_buffer: for i in 0 to N_ASICS_TOTAL-1 generate begin

u_elastic_buffer : entity work.mutrig_store
port map(
    i_clk_deser         => s_receivers_usrclk,
    i_clk_rd            => i_clk_core,
    i_reset             => i_rst_rx,
    i_aclear            => i_rst_core,
    i_event_data        => s_event_data(i),
    i_event_ready       => s_event_ready(i),
    i_new_frame         => s_new_frame(i),
    i_frame_info_rdy    => s_frame_info_rdy(i),
    i_end_of_frame      => s_end_of_frame(i),
    i_frame_info        => s_frame_info(i),
    i_frame_number      => s_frame_number(i),
    i_crc_error         => s_crc_error(i),
    --event data output inteface
    o_fifo_data         => s_fifos_data(i),
    o_fifo_empty        => s_fifos_empty(i),
    i_fifo_rd           => s_fifos_rd(i),
    --monitoring
    o_fifo_full         => s_fifos_full(i),
    i_reset_counters    => not s_SC_reset_counters_125_n,
    o_eventcounter      => s_eventcounter(i),
    o_timecounter       => s_timecounter(i),
    o_crcerrorcounter   => s_crcerrorcounter(i),
    o_framecounter      => s_framecounter(i),
    o_prbs_wrd_cnt      => s_prbs_wrd_cnt(i),
    o_prbs_err_cnt      => s_prbs_err_cnt(i),

    i_SC_mask           => i_SC_mask_rx(i)
);
end generate;

p_counterselect: process (s_timecounter, s_eventcounter, s_crcerrorcounter, s_framecounter, s_prbs_err_cnt, s_prbs_wrd_cnt, s_receivers_errorcounter,s_receivers_runcounter, s_receivers_synclosscounter, i_SC_counterselect)
begin
    o_counter_numerator<=(others =>'0');
    o_counter_denominator_high<=(others =>'0');
    o_counter_denominator_low<=(others =>'0');

    for i in 0 to N_ASICS_TOTAL-1 loop
        case i_SC_counterselect(2 downto 0) is
            when "000" =>
                if ( unsigned(i_SC_counterselect(6 downto 3)) = i ) then
                    o_counter_numerator <= s_eventcounter(i);
                end if;
                --opt: always use first
                o_counter_denominator_high <= s_timecounter(0)(63 downto 32);
                o_counter_denominator_low  <= s_timecounter(0)(31 downto  0);
            when "001" => 
                if ( unsigned(i_SC_counterselect(6 downto 3)) = i ) then
                    o_counter_numerator <= s_crcerrorcounter(i);
                    o_counter_denominator_high <= s_framecounter(i)(63 downto 32);
                    o_counter_denominator_low  <= s_framecounter(i)(31 downto  0);
                end if;
            when "010" =>
                if ( unsigned(i_SC_counterselect(6 downto 3)) = i ) then
                    o_counter_numerator <= s_prbs_err_cnt(i);
                    o_counter_denominator_high <= s_prbs_wrd_cnt(i)(63 downto 32);
                    o_counter_denominator_low  <= s_prbs_wrd_cnt(i)(31 downto  0);
                end if;
            when "011" =>
                if ( unsigned(i_SC_counterselect(6 downto 3)) = i ) then
                    o_counter_numerator <= s_receivers_errorcounter(i);
                    o_counter_denominator_high <= (others =>'0');
                    o_counter_denominator_low  <= s_receivers_runcounter(i);
                end if;
            when "100" =>
                if ( unsigned(i_SC_counterselect(6 downto 3)) = i ) then
                    o_counter_numerator <= s_receivers_synclosscounter(i);
                    o_counter_denominator_high <= (others =>'0');
                    o_counter_denominator_low <= (others =>'0');
                end if;

            when others =>
        end case;
    end loop;
end process;


--mux between asic channels
u_mux_A : entity work.framebuilder_mux
    generic map( 
        N_INPUTS => N_ASICS,
        N_INPUTID_BITS => 4,
        C_CHANNELNO_PREFIX => C_CHANNELNO_PREFIX_A
    )
    port map(
        i_coreclk           => i_clk_core,
        i_rst               => i_rst_core,
        i_timestamp_clk     => i_ts_clk,
        i_timestamp_rst     => i_ts_rst,
    --event data inputs interface
        i_source_data       => s_fifos_data(N_ASICS-1 downto 0),
        i_source_empty      => s_fifos_empty(N_ASICS-1 downto 0),
        o_source_rd         => s_fifos_rd(N_ASICS-1 downto 0),
    --event data output interface to big buffer storage
        o_sink_data         => s_A_buf_predec_data,
        i_sink_full         => s_A_buf_predec_full,
        o_sink_wr           => s_A_buf_predec_wr,
    --monitoring, errors, slow control
        o_busy              => s_A_mux_busy,
        o_sync_error        => o_frame_desync(0),
        i_SC_mask           => i_SC_mask(N_ASICS-1 downto 0),
        i_SC_nomerge        => '0'--,
    );

gen_dual_mux : if( N_MODULES > 1 ) generate
u_mux_B : entity work.framebuilder_mux
    generic map( 
        N_INPUTS => N_ASICS,
        N_INPUTID_BITS => 4,
        C_CHANNELNO_PREFIX => C_CHANNELNO_PREFIX_B
    )
    port map(
        i_coreclk       => i_clk_core,
        i_rst           => i_rst_core,
        i_timestamp_clk => i_ts_clk,
        i_timestamp_rst => i_ts_rst,
    --event data inputs interface
            i_source_data   => s_fifos_data(N_ASICS_TOTAL-1 downto N_ASICS),
        i_source_empty      => s_fifos_empty(N_ASICS_TOTAL-1 downto N_ASICS),
        o_source_rd         => s_fifos_rd(N_ASICS_TOTAL-1 downto N_ASICS),
    --event data output interface to big buffer storage
        o_sink_data     => s_B_buf_predec_data,
        i_sink_full     => s_B_buf_predec_full,
        o_sink_wr       => s_B_buf_predec_wr,
    --monitoring, errors, slow control
        o_busy          => s_B_mux_busy,
        o_sync_error    => o_frame_desync(1),
        i_SC_mask       => i_SC_mask(N_ASICS_TOTAL-1 downto N_ASICS),
        i_SC_nomerge    => '0'--,
    );

end generate;

--prbs decoder (two-stream)
u_decoder : entity work.prbs_decoder
    port map (
        i_coreclk       => i_clk_core,
        i_rst           => i_rst_core,

        i_A_data        => s_A_buf_predec_data,
        i_A_valid       => s_A_buf_predec_wr,
        i_B_data        => s_B_buf_predec_data,
        i_B_valid       => s_B_buf_predec_wr,

        o_A_data        => s_A_buf_data,
        o_A_valid       => s_A_buf_wr,
        o_B_data        => s_B_buf_data,
        o_B_valid       => s_B_buf_wr,
        i_SC_disable_dec=> i_SC_disable_dec
    );
    
-- generate lapse counter A
e_lapse_counter_A : entity work.lapse_counter
generic map ( N_CC => N_CC )
port map ( i_clk => i_ts_clk, i_reset_n => not i_ts_rst, i_CC => s_A_buf_data(20 downto 6),
    i_en => i_en_lapse_counter, i_upper_bnd => i_upper_bnd, o_CC => CC_corrected_A, o_cnt => open );

-- generate lapse counter B
e_lapse_counter_B : entity work.lapse_counter
generic map ( N_CC => N_CC )
port map ( i_clk => i_ts_clk, i_reset_n => not i_ts_rst, i_CC => s_B_buf_data(20 downto 6),
    i_en => i_en_lapse_counter, i_upper_bnd => i_upper_bnd, o_CC => CC_corrected_B, o_cnt => open );

--to common fifo buffer:
o_fifo_wr(0)                    <= s_A_buf_wr;
o_fifo_data(35 downto 0)        <=  "00" & s_A_buf_data(33 downto 21) & CC_corrected_A(14 downto 0) & s_A_buf_data(5 downto 0) when s_A_buf_data(33 downto 32) = "00" else
                                    "00" & s_A_buf_data;
s_A_buf_predec_full             <= i_common_fifos_almost_full(0);

gen_dual_cfifo: if( N_LINKS > 1 ) generate
    o_fifo_wr(1)                <= s_B_buf_wr;
    o_fifo_data(71 downto 36)   <=  "00" & s_B_buf_data(33 downto 21) & CC_corrected_B(14 downto 0) & s_B_buf_data(5 downto 0) when s_B_buf_data(33 downto 32) = "00" else
                                    "00" & s_B_buf_data;
    s_B_buf_predec_full         <= i_common_fifos_almost_full(1);
end generate;

nogen_dual: if( N_MODULES = 1 ) generate
    o_frame_desync(1)   <='0';
    s_B_buf_predec_wr   <='0';
end generate;

p_RC_all_done: process (i_clk_core)
begin
    if rising_edge(i_clk_core) then
        if(
            s_any_framegen_busy_156='0' and
            s_fifos_empty=(s_fifos_empty'range => '1') and
            s_A_mux_busy='0' and s_B_mux_busy='0' and
            s_A_buf_wr='0' and s_B_buf_wr='0'
        ) then
            o_RC_all_done <='1';
        else
            o_RC_all_done <= '0';
        end if;
    end if;
end process;

end architecture RTL;
