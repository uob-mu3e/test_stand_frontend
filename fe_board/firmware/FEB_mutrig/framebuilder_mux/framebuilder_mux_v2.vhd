---------------------------------------
--
-- Multiplexer for data from different asics, to be connected after channel buffer fifo (mutrig_store/datachannel)
-- Fair round robin arbitration for hit data, frame headers and trailers are combined and replaced by a global header and trailer
-- Marius Köppel April 2022
-- 
-- mkoeppel@uni-mainz.de
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

use work.mutrig.all;
use work.mudaq.all;


entity framebuilder_mux_v2 is
generic(
    -- num of input links
    N_INPUTS : integer;
    -- log2(N_INPUTS)
    N_INPUTS_INDEX : integer;
    --use prefix value as the first bits (MSBs) of the chip number field. Leave empty to append nothing and use all bits from Input # numbering
    C_ASICNO_PREFIX : std_logic_vector := ""--;
);
port (
    --event data inputs interface
    i_data          : in  mutrig_evtdata_array_t(N_INPUTS-1 downto 0);
    i_rempty        : in  std_logic_vector(N_INPUTS-1 downto 0);
    o_ren           : out std_logic_vector(N_INPUTS-1 downto 0);
    --event data output interface to big buffer storage
    o_data          : out std_logic_vector(33 downto 0);
    i_wfull         : in  std_logic;
    o_wen           : out std_logic;
    --still data to process. Does not check packet state, only if there is data in the chain.
    o_busy          : out std_logic;

    --monitoring, write-when-fill is prevented internally
    o_sync_error    : out std_logic;
    o_cc_diff       : out std_logic_vector(14 downto 0);
    -- mask input of asic
    i_mask          : in  std_logic_vector(N_INPUTS-1 downto 0);

    -- reset / clk
    i_ts_reset_n    : in  std_logic;

    i_clk           : in  std_logic;
    i_reset_n       : in  std_logic--;
);
end entity;

architecture rtl of framebuilder_mux_v2 is

    -- intput-data based combinatorics
    signal l_all_header, l_all_trailer, l_all_header_trailer : std_logic;
    signal l_header, l_trailer, l_crc_err, l_asic_drop, l_asic_bad_trailer, l_asic_over, l_frameid_nonsync_all : std_logic_vector(N_INPUTS-1 downto 0);

    -- combining header, frame numbers do not match
    signal l_frameid_nonsync      : std_logic; 
    signal l_any_crc_err          : std_logic;
    signal l_any_asic_overflow    : std_logic;
    signal l_any_asic_hitdropped  : std_logic;
    signal l_any_asic_bad_trailer : std_logic;

    -- select first non-masked data input for retreiving header and trailer 
    signal l_common_data : std_logic_vector(55 downto 0);

    -- global timestamp
    signal s_global_timestamp : std_logic_vector(47 downto 0);

    -- readout logic
    constant IDLE   : std_logic_vector(3 downto 0) := x"1";
    constant HEADER : std_logic_vector(3 downto 0) := x"2";
    constant T1     : std_logic_vector(3 downto 0) := x"3";
    constant HIT    : std_logic_vector(3 downto 0) := x"4";
    constant WAITING: std_logic_vector(3 downto 0) := x"5";
    constant TRAILER: std_logic_vector(3 downto 0) := x"6";
    constant HATR   : std_logic_vector(3 downto 0) := x"7";
    signal rd_state, rd_state_last : std_logic_vector(3 downto 0) := IDLE;

    -- output words
    signal w_t0, w_t1, w_trailer, w_hit : std_logic_vector(33 downto 0);
    signal s_sel_data : std_logic_vector(55 downto 0);
    signal s_asicnum : std_logic_vector(3 downto 0);
    signal ren : std_logic_vector(N_INPUTS-1 downto 0);

    -- one hot signal indicating current active input stream
    signal index, index_last : std_logic_vector(N_INPUTS-1 downto 0) := (0 => '1', others => '0');
    signal last_hit_type : std_logic_vector(1 downto 0);
    signal s_sel_index : natural;

    -- PLL test value
    signal cc_prev : std_logic_vector(14 downto 0);

begin

    --! input erros
    assert ( true
        and ( not ( N_INPUTS =  4 and N_INPUTS_INDEX = 2 ) or
              not ( N_INPUTS = 16 and N_INPUTS_INDEX = 4 ) )
    ) report "framebuilder_mux_v2"
        & "N_INPUTS_INDEX needs to be sqrt(N_INPUTS) see work.util.one_hot_to_index"
    severity failure;

    --! generate busy signal for run end
    o_busy <= '1' when unsigned(i_rempty) /= 0 and rd_state /= IDLE else '0';

    --! global timestamp generation
    process(i_clk, i_ts_reset_n)
    begin
    if ( i_ts_reset_n /= '1' ) then
        s_global_timestamp <= (others => '0');
    elsif rising_edge(i_clk) then
        s_global_timestamp <= s_global_timestamp + '1';
    end if;
    end process;

    --! source data inspection: all trailer, all header, hit requests.
    --! define signals l_all_header, l_all_trailer, l_request, common data (l_common_data, l_any_crc_err, l_any_asic_*, l_frameid_nonsync)
    gen_header_trailer : FOR i in 0 to N_INPUTS - 1 GENERATE
        l_header(i)             <= '1' when (i_data(i)(51 downto 50) = "10" and i_rempty(i) = '0') or i_mask(i) = '1' else '0';
        l_trailer(i)            <= '1' when (i_data(i)(51 downto 50) = "11" and i_rempty(i) = '0') or i_mask(i) = '1' else '0';
        l_crc_err(i)            <= '1' when i_mask(i) = '0' and i_data(i)(16) = '1' else '0';
        l_asic_over(i)          <= '1' when i_mask(i) = '0' and i_data(i)(17) = '1' else '0';
        l_asic_drop(i)          <= '1' when i_mask(i) = '0' and i_data(i)(18) = '1' else '0';
        l_asic_bad_trailer(i)   <= '1' when i_mask(i) = '0' and i_data(i)(19) = '1' else '0';
        -- check frameID
        l_frameid_nonsync_all(i) <= '1' when l_all_header = '1' and i_mask(i) = '0' and i_data(i)(15 downto 0) /= l_common_data(15 downto 0) else '0';
    END GENERATE;
    l_all_header            <=  '1' when work.util.and_reduce(l_header) = '1' else
                                '0';
    l_all_trailer           <=  '1' when work.util.and_reduce(l_trailer) = '1' else
                                '0';
    -- FIXME (MK): this is bad here should work without
    l_all_header_trailer    <=  '1' when work.util.and_reduce(l_header or l_trailer) = '1' else
                                '0';

    -- common data take the first non mask one
    l_common_data           <=  i_data(0) when i_mask(0) = '0' else
                                i_data(1) when i_mask(1) = '0' else
                                i_data(2) when i_mask(2) = '0' else
                                i_data(3) when i_mask(3) = '0';
    l_any_crc_err           <= '1' when work.util.or_reduce(l_crc_err) = '1' else '0';
    l_any_asic_overflow     <= '1' when work.util.or_reduce(l_asic_over) = '1' else '0';
    l_any_asic_hitdropped   <= '1' when work.util.or_reduce(l_asic_drop) = '1' else '0';
    l_any_asic_bad_trailer  <= '1' when work.util.or_reduce(l_asic_bad_trailer) = '1' else '0';
    l_frameid_nonsync       <= '1' when work.util.or_reduce(l_frameid_nonsync_all) = '1' else '0';

    -- readout state
    rd_state <= IDLE    when i_reset_n /= '1' else
                WAITING when i_wfull = '1' else
                HEADER  when l_all_header = '1' and (rd_state_last = IDLE or rd_state_last = TRAILER or rd_state_last = HATR) else
                T1      when rd_state_last = HEADER else
                TRAILER when l_all_trailer = '1' and (rd_state_last = T1 or rd_state_last = HIT) else
                HATR    when l_all_header_trailer = '1' and (rd_state_last = T1 or rd_state_last = HIT) else
                HIT     when rd_state_last = T1 or rd_state_last = HIT else
                IDLE    when l_all_header = '0' else
                WAITING;

    -- generate write signal
    o_wen <= '1' when rd_state = HEADER or rd_state = T1 or rd_state = TRAILER or rd_state = HATR else
             '1' when work.util.or_reduce(ren) = '1' and rd_state = HIT else
             '0';

    -- generate read signal
             -- do not read when we are in reset or the output is full
    ren   <= (others => '0') when i_wfull = '1' or i_reset_n /= '1' or rd_state = WAITING else
             -- read when when we are in state header or trailer
             (others => '1') when rd_state = HEADER or rd_state = TRAILER else
             -- read the once with a trailer
             l_trailer       when rd_state = HATR else
             -- read from inputs which dont have a header
             not l_header    when rd_state = IDLE else
             -- read from the current merged asic
             index and (not (i_rempty or i_mask or l_header or l_trailer)) when rd_state = HIT else
             (others => '0');
    o_ren <= ren;

    -- generate output data
    -- header word
    w_t0(33 downto 32)  <= MERGER_FIFO_PAKET_START_MARKER(1 downto 0);  --identifier (type header)
    w_t0(31 downto 0)   <= s_global_timestamp(47 downto 16);            --global timestamp
    -- t1 word
    w_t1(33 downto 32)      <= MERGER_FIFO_PAKET_T1_MARKER(1 downto 0); --identifier for t1
    w_t1(31 downto 16)      <= s_global_timestamp(15 downto 0);         --global timestamp
    w_t1(15)                <= l_frameid_nonsync;                       --frameID nonsync
    w_t1(14 downto 0)       <= l_common_data(14 downto 0);              --frameID
    -- trailer word
    w_trailer(33 downto 32)  <= MERGER_FIFO_PAKET_END_MARKER(1 downto 0);   --identifier for trailer
    w_trailer(31 downto 5)   <= (others => '0');                            --filler
    w_trailer(4)             <= '1' when rd_state = HATR else '0';          --mark header/trailer state
    w_trailer(3)             <= l_any_asic_bad_trailer;                     --asic package had a bad trailer
    w_trailer(2)             <= l_any_asic_hitdropped;                      --fpga fifo overflow flag
    w_trailer(1)             <= l_any_asic_overflow;                        --asic fifo overflow flag
    w_trailer(0)             <= l_any_crc_err;                              --crc error flag
    -- hit word
    w_hit(33 downto 32) <= "00";                        --identifier: data T part
    w_hit(31 downto 28) <= s_asicnum;                   -- asic number
    w_hit(27)           <= s_sel_data(48);              -- type (1=TPART, 0=EPART)
    w_hit(26 downto 22) <= s_sel_data(47 downto 43);    -- event data: chnum
    w_hit(21 downto 0)  <= s_sel_data(42 downto 21) when s_sel_data(48) = '0' else --T event data: ttime, eflag
                           s_sel_data(20 downto 0) & s_sel_data(21);               --E event data: etime, eflag(redun)

    o_data <=   w_t0 when rd_state = HEADER else
                w_t1 when rd_state = T1 else
                w_trailer when rd_state = TRAILER or rd_state = HATR else
                w_hit;

    --! select data from current index
    s_sel_index<= work.util.one_hot_to_index(index, N_INPUTS, N_INPUTS_INDEX);
    s_sel_data <= i_data(s_sel_index);
    s_asicnum  <=   C_ASICNO_PREFIX & "00" when s_sel_index = 0 else
                    C_ASICNO_PREFIX & "01" when s_sel_index = 1 else
                    C_ASICNO_PREFIX & "10" when s_sel_index = 2 else
                    C_ASICNO_PREFIX & "11" when s_sel_index = 3 else
                    (others => '0');

    -- TODO: add long mode index_last when last_hit_tpart = '0' and rd_state = HIT else
    index <=    work.util.round_robin_next(index_last, not (i_rempty or i_mask or l_header or l_trailer)) when rd_state = HIT and last_hit_type = "00" else
                index_last when rd_state = HIT and last_hit_type = "01" else -- last hit was epart so we read now the tpart
                (0 => '1', others => '0');

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        index_last      <= (0 => '1', others => '0');
        last_hit_type   <= (others => '0');
        rd_state_last   <= IDLE;
        cc_prev         <= (others => '0');
        o_cc_diff       <= (others => '0');
        --
    elsif rising_edge(i_clk) then
        index_last <= index;
        if ( rd_state = HIT ) then
            if ( last_hit_type = "01" and i_rempty(s_sel_index) = '0' ) then
                last_hit_type  <= (others => '0');
            -- we have a epart hit
            elsif ( last_hit_type = "00" and s_sel_data(48) = '0') then
                last_hit_type <= "01";
            end if;
        end if;
        if ( rd_state /= WAITING ) then
            rd_state_last <= rd_state;
        end if;
        if ( rd_state = HIT and rd_state_last = HIT ) then
            cc_prev     <= w_hit(20 downto 6);
            o_cc_diff   <= cc_prev - w_hit(20 downto 6);
        end if;
        --
    end if;
    end process;

end architecture;
