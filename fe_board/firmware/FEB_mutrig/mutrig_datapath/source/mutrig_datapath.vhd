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
    IS_SCITILE          : std_logic := '1';
    N_MODULES           : integer range 1 to 2 := 1;
    N_INPUTSRX          : positive := 8;
    N_ASICS             : positive := 1;
    N_LINKS             : positive := 1;
    N_CC                : positive := 15; -- will be always 15
    LVDS_PLL_FREQ       : real := 125.0;
    LVDS_DATA_RATE      : real := 1250.0;
    GEN_DUMMIES         : boolean := TRUE;
    INPUT_SIGNFLIP      : std_logic_vector(31 downto 0):=x"00000000";
    C_ASICNO_PREFIX_A   : std_logic_vector:=""; --use prefix value as the first bits (MSBs) of the chip number field. Leave empty to append nothing and use all bits from Input # numbering
    C_ASICNO_PREFIX_B   : std_logic_vector:=""--;
    --(e.g. Tiles,  one module with up to 16 ASICs, PREFIX="")
    --(e.g. Fibers, two modules with up to 4 ASICs each, PREFIX="00" ; "01" for A and B )
);
port (
    i_rst_core                  : in  std_logic;    -- logic reset of digital core (buffer clear, 156MHz clock synced)
    i_rst_rx                    : in  std_logic;    -- logic reset of lvds receivers (125MHz clock synced)
    i_stic_txd                  : in  std_logic_vector(N_INPUTSRX-1 downto 0);   -- serial data
    i_refclk_125_A              : in  std_logic;    -- ref clk for lvds pll (A-Side)
    i_refclk_125_B              : in  std_logic;    -- ref clk for lvds pll (B-Side)
    i_ts_clk                    : in  std_logic;    -- ref clk for global timestamp
    i_ts_rst                    : in  std_logic;    -- global timestamp reset, high active

    -- interface to asic fifos
    o_fifo_data                 : out std_logic_vector(N_LINKS*36-1 downto 0);
    o_fifo_wr                   : out std_logic_vector(N_LINKS-1 downto 0);
    i_common_fifos_almost_full  : in  std_logic_vector(N_LINKS-1 downto 0);

    -- slow control
    i_SC_disable_dec            : in  std_logic;
    i_SC_mask                   : in  std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
    i_SC_mask_rx                : in  std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
    i_SC_datagen_enable         : in  std_logic;
    i_SC_datagen_shortmode      : in  std_logic;
    i_SC_datagen_count          : in  std_logic_vector(9 downto 0);
    i_SC_rx_wait_for_all        : in  std_logic;
    i_SC_rx_wait_for_all_sticky : in  std_logic;
    i_link_data_reg             : in  std_logic_vector(31 downto 0);
    o_ch_rate                   : out work.util.slv32_array_t(127 downto 0);
    
    -- run control
    i_RC_may_generate           : in  std_logic; -- do not generate new frames for runstates that are not RUNNING, allows to let fifos run empty
    o_RC_all_done               : out std_logic; -- all fifos empty, all data read
    i_enable_length             : in  std_logic; -- enable to replace the length

    -- lapse lapse counter
    i_en_lapse_counter          : in  std_logic;
    i_upper_bnd                 : in  std_logic_vector(N_CC - 1 downto 0);
    i_lower_bnd                 : in  std_logic_vector(N_CC - 1 downto 0);

    -- monitors
    o_receivers_pll_lock        : out std_logic; -- pll lock flag
    o_receivers_dpa_lock        : out std_logic_vector(N_MODULES*N_ASICS-1 downto 0); -- dpa lock flag per channel
    o_receivers_ready           : out std_logic_vector(N_MODULES*N_ASICS-1 downto 0); -- receiver output ready flag
    o_frame_desync              : out std_logic_vector(1 downto 0);
    o_cc_diff                   : out std_logic_vector(14 downto 0);
    
    -- simulation input
    i_enablesim                 : in  std_logic := '0';
    i_simdata                   : in  std_logic_vector(8*N_MODULES * N_ASICS-1 downto 0) := (others => '0');
    i_simdatak                  : in  std_logic_vector(N_MODULES * N_ASICS-1 downto 0) := (others => '0');

    i_SC_reset_counters         : in  std_logic; --synchronous to i_clk_core
    o_fifos_full                : out std_logic_vector(N_MODULES*N_ASICS-1 downto 0); -- mutrig store fifo full
    o_counters                  : out work.util.slv32_array_t(10 * N_MODULES*N_ASICS-1 downto 0);

    i_reset_156_n               : in  std_logic;
    i_clk_156                   : in  std_logic;
    i_reset_125_n               : in  std_logic;
    i_clk_125                   : in  std_logic--;
);
end entity;

architecture rtl of mutrig_datapath is

    constant N_ASICS_TOTAL : natural := N_MODULES * N_ASICS;
    constant asicnum : std_logic_vector(4*8-1 downto 0) := x"76543210";

    -- TODO: add this to a header file
    subtype t_vector is std_logic_vector(N_ASICS_TOTAL-1 downto 0);
    type t_array_64b is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(64-1 downto 0);
    type t_array_48b is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(48-1 downto 0);
    subtype t_array_32b is work.util.slv32_array_t(N_ASICS_TOTAL-1 downto 0);
    type t_array_16b is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(16-1 downto 0);
    type t_array_8b  is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(8-1 downto 0);
    type t_array_2b  is array (N_ASICS_TOTAL-1 downto 0) of std_logic_vector(2-1 downto 0);

    -- serdes-frame_rcv
    signal s_receivers_state        : std_logic_vector(2*N_ASICS_TOTAL-1 downto 0);
    signal s_receivers_ready        : t_vector;
    signal s_receivers_data, s_receivers_data_length, s_receivers_data_reg : std_logic_vector(8*N_ASICS_TOTAL-1 downto 0);
    signal s_receivers_data_isk, s_receivers_data_isk_length, s_receivers_data_isk_reg : t_vector;
    signal s_receivers_all_ready    : std_logic;
    signal s_receivers_block        : std_logic;

    -- frame_rcv/datagen - fifo: fifo side, frame-receiver side, dummy datagenerator side
    signal s_crc_error: t_vector;
    signal s_frame_number,   s_rec_frame_number      : t_array_16b;
    signal s_frame_info,     s_rec_frame_info        : t_array_16b;
    signal s_new_frame,      s_rec_new_frame         : t_vector;
    signal s_frame_info_rdy, s_rec_frame_info_rdy    : t_vector;
    signal s_event_data,     s_rec_event_data        : t_array_48b;
    signal s_event_ready,    s_rec_event_ready       : t_vector;
    signal s_end_of_frame,   s_rec_end_of_frame      : t_vector;
    signal s_frec_busy                               : t_vector;
    signal s_any_framegen_busy : std_logic;

    -- data generator
    signal s_gen_event_data : std_logic_vector(47 downto 0);
    signal s_gen_frame_number, s_gen_frame_info : std_logic_vector(15 downto 0);
    signal s_gen_event_ready, s_gen_end_of_frame, s_gen_new_frame, s_gen_frame_info_rdy, s_gen_busy : std_logic;

    --fifo - frame collector mux
    signal s_fifos_empty    : std_logic_vector(N_ASICS_TOTAL-1 downto 0):=(others =>'1');
    signal s_fifos_data     : mutrig_evtdata_array_t(N_ASICS_TOTAL-1 downto 0);
    signal s_fifos_rd       : std_logic_vector(N_ASICS_TOTAL-1 downto 0);

    -- frame data for mux
    signal s_buf_data, s_buf_q : work.util.slv34_array_t(N_ASICS_TOTAL-1 downto 0);
    signal out_fifo_full : std_logic_vector(1 downto 0);
    signal s_buf_we, s_buf_full, s_buf_re, s_buf_empty, s_buf_sop, s_buf_eop, rempty : std_logic_vector(N_ASICS_TOTAL-1 downto 0);

    -- frame collector mux - prbs decoder
    signal s_A_mux_busy, s_B_mux_busy   : std_logic :='0';
    signal s_A_buf_predec_data          : std_logic_vector(33 downto 0);
    signal s_A_buf_predec_full          : std_logic;
    signal s_A_buf_predec_wr            : std_logic;
    signal s_B_buf_predec_data          : std_logic_vector(33 downto 0);
    signal s_B_buf_predec_full          : std_logic :='0';
    signal s_B_buf_predec_wr            : std_logic;

    -- prbs decoder - mu3edataformat-writer - common fifo
    signal s_A_buf_data     : std_logic_vector(33 downto 0);
    signal s_A_buf_data_reg : std_logic_vector(33 downto 0);
    signal s_A_buf_wr       : std_logic;
    signal s_B_buf_data     : std_logic_vector(33 downto 0);
    signal s_B_buf_data_reg : std_logic_vector(33 downto 0);
    signal s_B_buf_wr       : std_logic;

    -- readout link directly
    signal link_data : std_logic_vector(33 downto 0);
    signal link_en : std_logic;

    -- Scifi Counters per ASIC N_ASICS_TOTAL
    -- mutrig store:
    --  0: s_eventcounter
    --  1: s_timecounter low
    --  2: s_timecounter high
    --  3: s_crcerrorcounter
    --  4: s_framecounter
    --  5: s_prbs_wrd_cnt
    --  6: s_prbs_err_cnt
    -- rx
    --  7: s_receivers_runcounter
    --  8: s_receivers_errorcounter
    --  9: s_receivers_synclosscounter
    signal s_fifos_full                 : t_vector;
    signal s_eventcounter               : t_array_32b;
    signal s_timecounter                : t_array_64b;
    signal s_crcerrorcounter            : t_array_32b;
    signal s_framecounter               : t_array_64b;
    signal s_prbs_wrd_cnt               : t_array_64b;
    signal s_prbs_err_cnt               : t_array_32b;
    signal s_receivers_runcounter       : t_array_32b;
    signal s_receivers_errorcounter     : t_array_32b;
    signal s_receivers_synclosscounter  : t_array_32b;
    signal s_SC_reset_counters_125_n    : std_logic;

    -- lapse counter signals
    signal CC_corrected_A : std_logic_vector(N_CC - 1 downto 0);
    signal CC_corrected_B : std_logic_vector(N_CC - 1 downto 0);

    -- signals for inputs / outputs
    signal fifo_data, sync_fifo_data        : std_logic_vector(N_LINKS*36-1 downto 0);
    signal fifo_we                          : std_logic_vector(N_LINKS-1 downto 0);
    signal RC_all_done                      : std_logic_vector(0 downto 0) := (others => '0');
    signal sync_fifo_read, sync_fifo_empty  : std_logic_vector(N_LINKS-1 downto 0);

begin

    o_fifos_full <= s_fifos_full;
    --! output counter
    gen_counters : for i in 0 to N_ASICS_TOTAL-1 generate
        o_counters(0+i*10) <= s_eventcounter(i);
        o_counters(1+i*10) <= s_timecounter(0)(31 downto 0); -- take only the first one
        o_counters(2+i*10) <= s_timecounter(0)(63 downto 32);
        o_counters(3+i*10) <= s_crcerrorcounter(i);
        o_counters(4+i*10) <= s_framecounter(i)(31 downto 0); -- we only take the low bits for now
        o_counters(5+i*10) <= s_prbs_wrd_cnt(i)(31 downto 0); -- we only take the low bits for now
        o_counters(6+i*10) <= s_prbs_err_cnt(i);
        o_counters(7+i*10) <= s_receivers_runcounter(i);
        o_counters(8+i*10) <= s_receivers_errorcounter(i);
        o_counters(9+i*10) <= s_receivers_synclosscounter(i);
    end generate;

    rst_sync_counter : entity work.reset_sync
    port map( i_reset_n => not i_SC_reset_counters, o_reset_n => s_SC_reset_counters_125_n, i_clk => i_clk_125);

    -- synthesis read_comments_as_HDL on
    -- u_rxdeser: entity work.receiver_block
    -- generic map(
    --     IS_SCITILE      => IS_SCITILE,
    --     NINPUT          => N_INPUTSRX,
    --     LVDS_PLL_FREQ   => LVDS_PLL_FREQ,
    --     LVDS_DATA_RATE  => LVDS_DATA_RATE,
    --     INPUT_SIGNFLIP  => INPUT_SIGNFLIP
    -- )
    -- port map(
    --     rx_in                                           => i_stic_txd,

    --     rx_state(2*N_ASICS_TOTAL-1 downto 0)            => s_receivers_state,
    --     rx_ready(N_ASICS_TOTAL-1 downto 0)              => s_receivers_ready,
    --     pll_locked                                      => o_receivers_pll_lock,
    --     rx_dpa_locked_out(N_ASICS_TOTAL-1 downto 0)     => o_receivers_dpa_lock,
    --     rx_runcounter(N_ASICS_TOTAL-1 downto 0)         => s_receivers_runcounter,
    --     rx_errorcounter(N_ASICS_TOTAL-1 downto 0)       => s_receivers_errorcounter,
    --     rx_synclosscounter(N_ASICS_TOTAL-1 downto 0)    => s_receivers_synclosscounter,
    --     reset_n_errcnt                                  => s_SC_reset_counters_125_n,

    --     o_rx_data(N_ASICS_TOTAL*8-1 downto 0)           => s_receivers_data,
    --     o_rx_datak(N_ASICS_TOTAL-1 downto 0)            => s_receivers_data_isk,

    --     rx_inclock_A        => i_refclk_125_A,
    --     rx_inclock_B        => i_refclk_125_B,

    --     i_reset_n           => not i_rst_rx,
    --     i_clk               => i_clk_125
    -- );
    -- synthesis read_comments_as_HDL off

    o_receivers_ready <= s_receivers_ready;

    -- generate a pll-synchronous all-ready signal for the data receivers.
    -- this assures all start dumping data into the fifos at the same time, and we do not enter a deadlock scenario from the start
    gen_ready_all : process (i_clk_125, i_rst_rx, s_receivers_ready, i_SC_mask_rx)
    variable v_ready : std_logic_vector(N_ASICS_TOTAL-1 downto 0);
    begin
    if ( i_rst_rx = '1' ) then
        s_receivers_all_ready <= '0';
        --
    elsif rising_edge(i_clk_125) then
        v_ready := s_receivers_ready or i_SC_mask_rx;
        if ( v_ready = ((v_ready'range)=>'1') ) then
            s_receivers_all_ready <= '1';
        end if;
        --
    end if;
    end process;

    -- if i_SC_rx_wait_for_all is set, wait for all (not masked) receivers to become ready before letting any data pass through the frame unpacking blocks.
    -- if i_SC_rx_wait_for_all_sticky is set in addition, the all_ready property is sticky: once all receivers become ready do not block data again.
    --            The sticky bit is cleared with i_reset
    --            Otherwise, data is blocked as soon as one receiver is loosing the pattern or sync.
    releasedata_p : process(i_clk_125, i_rst_rx)
    begin
    if rising_edge(i_clk_125) then
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

    -- data generator
    u_data_dummy : entity work.stic_dummy_data
    port map (
        i_reset             => i_rst_rx,
        i_clk               => i_clk_125,
        -- configuration
        i_enable            => i_SC_datagen_enable and i_RC_may_generate,
        i_fast              => i_SC_datagen_shortmode,
        i_cnt               => i_SC_datagen_count,
        -- to mutrig-store instance
        o_frame_number      => s_gen_frame_number,
        o_frame_info        => s_gen_frame_info,
        o_frame_info_rdy    => s_gen_frame_info_rdy,
        o_new_frame         => s_gen_new_frame,
        o_event_data        => s_gen_event_data,
        o_event_ready       => s_gen_event_ready,
        o_end_of_frame      => s_gen_end_of_frame,
        o_busy              => s_gen_busy--,
    );

    u_link_data : entity work.link_data
    generic map (
        N_ASICS_TOTAL => N_ASICS_TOTAL--,
    )
    port map (
        i_clk       => i_clk_125,
        i_reset_n   => not i_rst_rx,

        i_data      => s_receivers_data_reg,
        i_byteisk   => s_receivers_data_isk_reg,
        i_link      => i_link_data_reg(3 downto 0),

        o_data      => link_data,
        o_data_en   => link_en--,
    );

    gen_frame: for i in 0 to N_ASICS_TOTAL-1 generate begin
    
        process(i_clk_125, i_reset_125_n)
        begin
        if ( i_reset_125_n /= '1' ) then
            s_receivers_data_reg((i+1)*8-1 downto i*8) <= x"BC";
            s_receivers_data_isk_reg(i) <= '1';
            --
        elsif rising_edge(i_clk_125) then
            if ( i_enablesim = '0' ) then
                s_receivers_data_reg((i+1)*8-1 downto i*8) <= s_receivers_data((i+1)*8-1 downto i*8);
                s_receivers_data_isk_reg(i) <= s_receivers_data_isk(i);
            else
                s_receivers_data_reg((i+1)*8-1 downto i*8) <= i_simdata((i+1)*8-1 downto i*8);
                s_receivers_data_isk_reg(i) <= i_simdatak(i);
            end if;
            --
        end if;
        end process;

        u_replace_length : entity work.replace_length
        port map (
            i_reset_n => not i_rst_rx,
            i_clk     => i_clk_125,
            i_data    => s_receivers_data_reg((i+1)*8-1 downto i*8),
            i_datak   => s_receivers_data_isk_reg(i),
            i_enable  => i_enable_length,

            o_data    => s_receivers_data_length((i+1)*8-1 downto i*8),
            o_datak   => s_receivers_data_isk_length(i)--,
        );

        u_frame_rcv : entity work.frame_rcv
        generic map(
            EVENT_DATA_WIDTH        => 48,
            N_BYTES_PER_WORD        => 6,
            N_BYTES_PER_WORD_SHORT  => 3
        )
        port map (
            i_rst               => i_rst_rx,
            i_clk               => i_clk_125,
            i_data              => s_receivers_data_length((i+1)*8-1 downto i*8),
            i_byteisk           => s_receivers_data_isk_length(i),
            i_enable            => i_enablesim or (i_RC_may_generate and not (s_receivers_block or (not s_receivers_ready(i) and not i_SC_rx_wait_for_all))),

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

        -- multiplex between physical and generated data sent to the elastic buffers
        process(i_clk_125, i_reset_125_n)
        begin
        if ( i_reset_125_n /= '1' ) then
            s_frame_number(i)   <= (others => '0');
            s_frame_info(i)     <= (others => '0');
            s_event_data(i)     <= (others => '0');
            s_event_ready(i)    <= '0';
            s_frame_info_rdy(i) <= '0';
            s_new_frame(i)      <= '0';
            s_end_of_frame(i)   <= '0';
            --
        elsif ( rising_edge(i_clk_125) ) then
            -- use busy from datagenerator to ensure safe takeover
            if ( i_SC_datagen_enable = '1' or s_gen_busy = '1' ) then
                s_frame_number(i)   <= s_gen_frame_number;
                s_frame_info(i)     <= s_gen_frame_info;
                s_event_data(i)     <= s_gen_event_data;
                s_event_ready(i)    <= s_gen_event_ready;
                s_frame_info_rdy(i) <= s_gen_frame_info_rdy;
                s_new_frame(i)      <= s_gen_new_frame;
                s_end_of_frame(i)   <= s_gen_end_of_frame;
            else
                s_frame_number(i)   <= s_rec_frame_number(i);
                s_frame_info(i)     <= s_rec_frame_info(i);
                s_event_data(i)     <= s_rec_event_data(i);
                s_event_ready(i)    <= s_rec_event_ready(i);
                s_frame_info_rdy(i) <= s_rec_frame_info_rdy(i);
                s_new_frame(i)      <= s_rec_new_frame(i);
                s_end_of_frame(i)   <= s_rec_end_of_frame(i);
            end if;
        end if;
        end process;

    end generate;

    -- p_frec_busy_sync
    process(i_clk_125)
    begin
    if rising_edge(i_clk_125) then
        s_any_framegen_busy <= '0';
        if ( i_SC_datagen_enable = '1' and s_gen_busy /= '0' ) then
            s_any_framegen_busy <= '1';
        end if;
        if ( i_SC_datagen_enable='0' and unsigned(s_frec_busy) /= 0 ) then
            s_any_framegen_busy <= '1';
        end if;
    end if;
    end process;

    g_buffer: for i in 0 to N_ASICS_TOTAL-1 generate begin
        u_elastic_buffer : entity work.mutrig_store
        port map(
            i_clk_deser         => i_clk_125,
            i_clk_rd            => i_clk_125,
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
            -- event data output inteface
            o_fifo_data         => s_fifos_data(i),
            o_fifo_empty        => s_fifos_empty(i),
            i_fifo_rd           => s_fifos_rd(i),
            -- monitoring
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

        --use mux for creating format
        u_data_unpacker : entity work.data_unpacker
        generic map( 
            asicnum      => asicnum((i+1)*4 - 1 downto i*4)--,
        )
        port map(
            -- event data input
            i_data       => s_fifos_data(i),
            i_mask       => i_SC_mask(i),
            i_rempty     => s_fifos_empty(i),
            o_ren        => s_fifos_rd(i),

            -- event data output
            o_data       => s_buf_data(i),
            i_wfull      => s_buf_full(i),
            o_wen        => s_buf_we(i),

            i_clk        => i_clk_125,
            i_ts_reset_n => not i_ts_rst,
            i_reset_n    => not i_rst_core--,
        );

        e_channel_rate : entity work.ch_rate
        generic map(
            num_ch => 32
        )
        port map(
            i_hit       => s_buf_data(i),
            i_en        => s_buf_we(i),

            o_ch_rate   => o_ch_rate((i+1)*32 - 1 downto i*32),

            i_clk       => i_clk_125,
            i_reset_n   => i_reset_125_n--,
        );

        u_mux_fifo : entity work.ip_dcfifo_v2
        generic map (
            g_ADDR_WIDTH => 10,
            g_DATA_WIDTH => 34,
            g_WREG_N => 1,
            g_RREG_N => 1--,
        )
        port map (
            i_wdata     => s_buf_data(i),
            i_we        => s_buf_we(i),
            o_wfull     => s_buf_full(i),
            i_wclk      => i_clk_125,

            o_rdata     => s_buf_q(i),
            i_rack      => s_buf_re(i),
            o_rempty    => s_buf_empty(i),
            i_rclk      => i_clk_125,

            i_reset_n   => not i_rst_core--,
        );

        s_buf_sop(i) <= '1' when s_buf_q(i)(33 downto 32) = "10" else '0';
        s_buf_eop(i) <= '1' when s_buf_q(i)(33 downto 32) = "11" else '0';

    end generate;

    --mux between asic channels
--    u_mux_A : entity work.framebuilder_mux_v2
--    generic map( 
--        N_INPUTS => N_ASICS,
--        -- NOTE: N_INPUTS_INDEX = log2(N_ASICS)
--        N_INPUTS_INDEX => 2,
--        C_ASICNO_PREFIX => C_ASICNO_PREFIX_A--,
--    )
--    port map(
--        -- event data inputs interface
--        i_data       => s_fifos_data(N_ASICS-1 downto 0),
--        i_rempty     => s_fifos_empty(N_ASICS-1 downto 0),
--        o_ren        => s_fifos_rd(N_ASICS-1 downto 0),
--        -- event data output interface to big buffer storage
--        o_data       => s_A_buf_predec_data,
--        i_wfull      => s_A_buf_predec_full,
--        o_wen        => s_A_buf_predec_wr,
--        -- monitoring, errors, slow control
--        o_busy       => s_A_mux_busy,
--        o_sync_error => o_frame_desync(0),
--        i_mask       => i_SC_mask(N_ASICS-1 downto 0),
--        -- reset / clk
--        i_ts_reset_n => not i_ts_rst,
--        i_clk        => i_clk_125,
--        i_reset_n    => not i_rst_core--,
--    );
    --mux between asic channels
--    u_mux_A : entity work.framebuilder_mux
--    generic map( 
--        N_INPUTS => N_ASICS,
--        N_INPUTID_BITS => 4,
--        C_CHANNELNO_PREFIX => C_ASICNO_PREFIX_A--,
--    )
--    port map(
--        i_coreclk           => i_clk_125,
--        i_rst               => i_rst_core,
--        i_timestamp_clk     => i_clk_125,
--        i_timestamp_rst     => i_ts_rst,
--        --event data inputs interface
--        i_source_data       => s_fifos_data(N_ASICS-1 downto 0),
--        i_source_empty      => s_fifos_empty(N_ASICS-1 downto 0),
--        o_source_rd         => s_fifos_rd(N_ASICS-1 downto 0),
--        --event data output interface to big buffer storage
--        o_sink_data         => s_A_buf_predec_data,
--        i_sink_full         => s_A_buf_predec_full,
--        o_sink_wr           => s_A_buf_predec_wr,
--        --monitoring, errors, slow control
--        o_busy              => s_A_mux_busy,
--        o_sync_error        => o_frame_desync(0),
--        i_SC_mask           => i_SC_mask(N_ASICS-1 downto 0),
--        i_SC_nomerge        => '0'--,
--    );

    rempty(3 downto 0) <= s_buf_empty(3 downto 0) or i_SC_mask(3 downto 0);
    -- round robin between asics
    e_stream_merger : entity work.stream_merger_vector
    generic map (
        g_set_type => false,
        N => 4--,
    )
    port map (
        -- input stream
        i_rdata     => s_buf_q(3 downto 0),
        i_sop       => s_buf_sop(3 downto 0),
        i_eop       => s_buf_eop(3 downto 0),
        i_rempty    => rempty(3 downto 0),
        o_rack      => s_buf_re(3 downto 0),

        -- output stream
        o_wdata     => s_A_buf_predec_data,
        i_wfull     => out_fifo_full(0),
        o_we        => s_A_buf_predec_wr,
        o_busy      => s_A_mux_busy,

        i_reset_n   => not i_rst_core,
        i_clk       => i_clk_125--,
    );

    gen_dual_mux : if( N_MODULES > 1 ) generate
        u_mux_B : entity work.framebuilder_mux_v2
        generic map(
            N_INPUTS => N_ASICS,
            -- NOTE: N_INPUTS_INDEX = sqrt(N_ASICS)
            N_INPUTS_INDEX => 2,
            C_ASICNO_PREFIX => C_ASICNO_PREFIX_B--,
        )
        port map(
            -- event data inputs interface
            i_data       => s_fifos_data(N_ASICS_TOTAL-1 downto N_ASICS),
            i_rempty     => s_fifos_empty(N_ASICS_TOTAL-1 downto N_ASICS),
            o_ren        => s_fifos_rd(N_ASICS_TOTAL-1 downto N_ASICS),
            -- event data output interface to big buffer storage
            o_data       => s_B_buf_predec_data,
            i_wfull      => s_B_buf_predec_full,
            o_wen        => s_B_buf_predec_wr,
            -- monitoring, errors, slow control
            o_busy       => s_B_mux_busy,
            o_sync_error => o_frame_desync(1),
            o_cc_diff    => o_cc_diff,
            i_mask       => i_SC_mask(N_ASICS_TOTAL-1 downto N_ASICS),
            -- reset / clk
            i_ts_reset_n => not i_ts_rst,
            i_clk        => i_clk_125,
            i_reset_n    => not i_rst_core--,
        );
        -- u_mux_B : entity work.framebuilder_mux
        -- generic map( 
        --     N_INPUTS => N_ASICS,
        --     N_INPUTID_BITS => 4,
        --     C_CHANNELNO_PREFIX => C_ASICNO_PREFIX_B--,
        -- )
        -- port map(
        --     i_coreclk           => i_clk_125,
        --     i_rst               => i_rst_core,
        --     i_timestamp_clk     => i_clk_125,
        --     i_timestamp_rst     => i_ts_rst,
        --     --event data inputs interface
        --     i_source_data       => s_fifos_data(N_ASICS_TOTAL-1 downto N_ASICS),
        --     i_source_empty      => s_fifos_empty(N_ASICS_TOTAL-1 downto N_ASICS),
        --     o_source_rd         => s_fifos_rd(N_ASICS_TOTAL-1 downto N_ASICS),
        --     --event data output interface to big buffer storage
        --     o_sink_data         => s_B_buf_predec_data,
        --     i_sink_full         => s_B_buf_predec_full,
        --     o_sink_wr           => s_B_buf_predec_wr,
        --     --monitoring, errors, slow control
        --     o_busy              => s_B_mux_busy,
        --     o_sync_error        => o_frame_desync(1),
        --     i_SC_mask           => i_SC_mask(N_ASICS_TOTAL-1 downto N_ASICS),
        --     i_SC_nomerge        => '0'--,
        -- );
    end generate;

    --prbs decoder (two-stream)
    u_decoder : entity work.prbs_decoder
    port map (
        i_coreclk       => i_clk_125,
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
        i_upper_bnd => i_upper_bnd, i_lower_bnd => i_lower_bnd, o_CC => CC_corrected_A, o_cnt => open );

    -- to fifo_out_1
    fifo_data(35 downto 0) <=   "00" & link_data when i_link_data_reg(31) = '1' else
                                "00" & s_A_buf_data(33 downto 21) & CC_corrected_A(14 downto 0) & s_A_buf_data(5 downto 0) when s_A_buf_data(33 downto 32) = "00" and i_en_lapse_counter = '1' else
                                "00" & s_A_buf_data;
    fifo_we(0)             <= link_en          when i_link_data_reg(31) = '1' else s_A_buf_wr;
        -- when i_en_lapse_counter = '0' else
        --"00" & s_A_buf_data(33 downto 21) & CC_corrected_A(14 downto 0) & s_A_buf_data(5 downto 0) when ( s_A_buf_data(33 downto 32) = "00" ) else
        --"00" & s_A_buf_data;

    e_fifo_out_1 : entity work.ip_dcfifo_v2
    generic map (
        g_ADDR_WIDTH => 8,
        g_DATA_WIDTH => 36--,
    )
    port map (
        i_wdata     => fifo_data(35 downto 0),
        i_we        => fifo_we(0),
        o_wfull     => out_fifo_full(0),
        i_wclk      => i_clk_125,

        o_rdata     => sync_fifo_data(35 downto 0),
        i_rack      => sync_fifo_read(0),
        o_rempty    => sync_fifo_empty(0),
        i_rclk      => i_clk_156,

        i_reset_n   => i_reset_156_n--,
    );

    o_fifo_data(35 downto 0)    <= sync_fifo_data(35 downto 0) when sync_fifo_empty(0) = '0' else (others => '0');
    sync_fifo_read(0)           <= '1' when sync_fifo_empty(0) = '0' else '0';
    o_fifo_wr(0)                <= '1' when sync_fifo_empty(0) = '0' else '0';

    e_sync_common_fifos_almost_full_A : entity work.ff_sync
    generic map ( W => i_common_fifos_almost_full'length )
    port map (
        i_d => i_common_fifos_almost_full, o_q(0) => s_A_buf_predec_full,
        i_reset_n => i_reset_125_n, i_clk => i_clk_125--,
    );

    gen_dual_cfifo: if( N_LINKS > 1 ) generate

        -- generate lapse counter B
        e_lapse_counter_B : entity work.lapse_counter
        generic map ( N_CC => N_CC )
        port map ( i_clk => i_ts_clk, i_reset_n => not i_ts_rst, i_CC => s_B_buf_data(20 downto 6),
            i_upper_bnd => i_upper_bnd, i_lower_bnd => i_lower_bnd, o_CC => CC_corrected_B, o_cnt => open );

        -- to fifo_out_2
        -- we just send the same data from the link directly on both fifos
        fifo_data(71 downto 36) <= "00" & link_data when i_link_data_reg(31) = '1' else "00" & s_B_buf_data;
        fifo_we(1)              <= link_en          when i_link_data_reg(31) = '1' else s_B_buf_wr;    
            -- when i_en_lapse_counter = '0' else
            --"00" & s_B_buf_data(33 downto 21) & CC_corrected_B(14 downto 0) & s_B_buf_data(5 downto 0) when ( s_B_buf_data(33 downto 32) = "00" ) else
            --"00" & s_B_buf_data;

        e_fifo_out_2 : entity work.ip_dcfifo_v2
        generic map (
            g_ADDR_WIDTH => 8,
            g_DATA_WIDTH => 36--,
        )
        port map (
            i_wdata     => fifo_data(71 downto 36),
            i_we        => fifo_we(1),
            o_wfull     => out_fifo_full(1),
            i_wclk      => i_clk_125,

            o_rdata     => sync_fifo_data(71 downto 36),
            i_rack      => sync_fifo_read(1),
            o_rempty_n  => sync_fifo_empty(1),
            i_rclk      => i_clk_156,

            i_reset_n   => i_reset_156_n--,
        );

        o_fifo_data(71 downto 36)   <= sync_fifo_data(71 downto 36) when sync_fifo_empty(1) = '0' and N_MODULES > 1 else (others => '0');
        sync_fifo_read(1)           <= '1' when sync_fifo_empty(1) = '0' and N_MODULES > 1 else '0';
        o_fifo_wr(1)                <= '1' when sync_fifo_empty(1) = '0' and N_MODULES > 1 else '0';

        e_sync_common_fifos_almost_full_B : entity work.ff_sync
        generic map ( W => i_common_fifos_almost_full'length )
        port map (
            i_d => i_common_fifos_almost_full, o_q(1) => s_B_buf_predec_full,
            i_reset_n => i_reset_125_n, i_clk => i_clk_125--,
        );
    end generate;

    nogen_dual: if( N_MODULES = 1 ) generate
        o_frame_desync(1)   <='0';
        s_B_buf_predec_wr   <='0';
    end generate;

    -- p_RC_all_done
    process(i_clk_125)
    begin
    if rising_edge(i_clk_125) then
        if(
            s_any_framegen_busy = '0'                       and
            s_fifos_empty = (s_fifos_empty'range => '1')    and
            s_A_mux_busy = '0'                              and
            s_B_mux_busy = '0'                              and
            s_A_buf_wr = '0'                                and
            s_B_buf_wr = '0'
        ) then
            RC_all_done(0) <='1';
        else
            RC_all_done(0) <= '0';
        end if;
    end if;
    end process;

    e_sync_RC_all_done : entity work.ff_sync
    generic map ( W => RC_all_done'length )
    port map (
        i_d => RC_all_done, o_q(0) => o_RC_all_done,
        i_reset_n => i_reset_156_n, i_clk => i_clk_156--,
    );

end architecture;
