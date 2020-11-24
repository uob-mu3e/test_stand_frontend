-- data generator to be insterted at the output of the Mupix hitsorter
-- Martin Mueller (muellem@uni-mainz.de)
-- November 2020

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;
use work.daq_constants.all;
use work.lfsr_taps.all;

entity mp_sorter_datagen is
port (
    i_reset_n           : in  std_logic;
    i_clk               : in  std_logic;
    i_running           : in  std_logic;
    i_global_ts         : in  std_logic_vector(63 downto 0);
    i_control_reg       : in  std_logic_vector(31 downto 0);
    o_hit_counter       : out std_logic_vector(63 downto 0);
    o_fifo_wdata        : out std_logic_vector(35 downto 0);
    o_fifo_write        : out std_logic;

    i_evil_register     : in  std_logic_vector(31 downto 0) := (others => '0');
    o_mischief_managed  : out std_logic--;
);
end entity;

architecture rtl of mp_sorter_datagen is

    signal reset                : std_logic;

    signal fwdata               : std_logic_vector(35 downto 0);
    signal fwrite               : std_logic;
    signal enable               : std_logic;
    signal running_prev         : std_logic;
    signal hit_counter          : unsigned(63 downto 0);
    signal frame_ts_overflow    : std_logic;
    signal packet_ts_overflow   : std_logic;
    signal run_shutdown         : std_logic;
    signal global_ts            : std_logic_vector(63 downto 0);

    type genstate_t   is (head1, head2, subhead, trail, EoR, hitgen, idle);
    signal genstate : genstate_t;

    -- rate control signals
    signal produce_next_packet  : std_logic;
    signal produce_next_frame   : std_logic;
    signal produce_next_hit     : std_logic;
    
    signal next_hit_p_range     : integer;

    -- control signals for evil actions against downstream components xD
    signal unsorted             : std_logic := '0'; -- send hits unsorted in time
    signal skip_frame           : std_logic := '0'; -- skip one subheader
    signal repeat_frame_ts      : std_logic := '0'; -- send subheader again/ to early
    signal data_burst           : std_logic := '0'; -- full steam for x cycles
    signal miss_trailer         : std_logic := '0'; -- skip trailer
    signal miss_header          : std_logic := '0'; -- skip header
    signal miss_half_of_header  : std_logic := '0'; -- skip 1/2 header
    signal miss_EOR             : std_logic := '0'; -- skip end of run
    signal complete_nonsense    : std_logic := '0'; -- random timestamp
    
    signal n_evil               : std_logic_vector(7 downto 0);
    signal n_evil_remaining     : std_logic_vector(7 downto 0);
    signal evil_probability     : std_logic_vector(7 downto 0);--something like and_reduce(random1(setting downto 0))
    signal mischief_managed     : std_logic; -- a single one

    -- randoms:
    signal ts                   : std_logic_vector(3 downto 0);
    signal chipID_index         : std_logic_vector(5 downto 0);
    signal chipID               : std_logic_vector(5 downto 0);
    signal row                  : std_logic_vector(7 downto 0);
    signal col                  : std_logic_vector(7 downto 0);
    signal tot                  : std_logic_vector(5 downto 0);

    -- 64 bit lfsr taps are not good for the rate distribution, using 65 bit instead
    signal random0              : std_logic_vector(64 downto 0);

    type valid_ID_t             is array (3 downto 0) of std_logic_vector(5 downto 0);
    constant valid_chipIDs      : valid_ID_t :=("001000", "010001", "100010", "000011"); --TODO: list all valid ID's depending on reg for layer

    begin
    o_fifo_wdata <= fwdata;
    o_fifo_write <= fwrite;
    enable       <= i_control_reg(31);
    o_hit_counter<= std_logic_vector(hit_counter);
    reset        <= not i_reset_n;

    next_hit_p_range <= to_integer(unsigned(i_control_reg(3 downto 0)));

    process(i_clk,i_reset_n)
    begin
        if ( i_reset_n = '0' ) then
            fwdata              <= (others => '0');
            fwrite              <= '0';
            run_shutdown        <= '0';
            produce_next_packet <= '0';
            produce_next_frame  <= '0';
            produce_next_hit    <= '0';
            hit_counter         <= (others => '0');
            running_prev        <= '0';
            frame_ts_overflow   <= '0';
            packet_ts_overflow  <= '0';
            genstate            <= idle;
            global_ts           <= (others => '0');

        elsif rising_edge(i_clk) then
            fwdata              <= (others => '0');
            fwrite              <= '0';
            running_prev        <= i_running;
            frame_ts_overflow   <= '0';
            packet_ts_overflow  <= '0';
            mischief_managed    <= '0';
            
            -------REMOVE / generate -------------------
            produce_next_packet <= '1';
            produce_next_frame  <= '1';
            ts                  <= "1111";
            ---------------------------------

            if(i_control_reg(4) = '0') then 
                produce_next_hit    <= '1';
            else
                produce_next_hit    <= and_reduce(random0(next_hit_p_range downto 0));
            end if;

            if(running_prev = '1' and i_running = '0') then 
                run_shutdown        <= '0';
            end if;

            if(i_global_ts(3 downto 0) = "1110") then
                frame_ts_overflow   <= '1';
            end if;

            if(i_global_ts(9 downto 0) = "1111111110") then
                packet_ts_overflow  <= '1';
            end if;

            if(complete_nonsense = '1') then
                global_ts <= random0(63 downto 0);
                mischief_managed <= '1';
            else
                global_ts <= i_global_ts;
            end if;

            case genstate is
                when idle =>
                    if(run_shutdown = '1') then 
                        genstate <= EoR;
                    elsif( i_running = '1' and enable = '1' and produce_next_packet = '1') then
                        if(miss_header = '0') then 
                            genstate <= head1;
                        else
                            genstate <= subhead;
                            mischief_managed <= '1';
                        end if;
                    end if;

                when head1 =>
                    if(miss_half_of_header = '0') then
                        genstate            <= head2;
                    else
                        genstate            <= subhead;
                        mischief_managed    <= '1';
                    end if;
                    fwdata(35 downto 32)    <= MERGER_FIFO_PAKET_START_MARKER;
                    fwdata(31 downto  0)    <= global_ts(47 downto 16);
                    fwrite                  <= '1';

                when head2 =>
                    genstate                <= subhead;
                    fwdata(31 downto 16)    <= global_ts(15 downto  0);
                    fwrite                  <= '1';

                when subhead =>
                    if(packet_ts_overflow = '1') then
                        genstate            <= trail;
                    elsif(produce_next_frame = '1') then 
                        genstate            <= hitgen;
                        fwdata(27 downto 22)<= "111111";
                        fwdata(21 downto 16)<= global_ts(9 downto 4);
                        fwrite              <= '1';
                    end if;

                when hitgen =>
                    if(packet_ts_overflow = '1') then
                        genstate            <= trail;
                    elsif(frame_ts_overflow = '1') then
                        genstate            <= subhead;
                    elsif(repeat_frame_ts = '1') then 
                        genstate            <= subhead;
                        mischief_managed    <= '1';
                    end if;
                    if(produce_next_hit = '1') then
                        fwdata              <= "0000" & ts & chipID & row & col & tot;
                        fwrite              <= '1';
                        hit_counter         <= hit_counter + 1;
                    end if;

                when trail =>
                    genstate                <= idle;
                    if(miss_trailer = '0') then 
                        fwdata(35 downto 32)<= MERGER_FIFO_PAKET_END_MARKER;
                        fwrite              <= '1';
                    else
                        mischief_managed    <= '1';
                    end if;

                when EoR =>
                    if(miss_EOR = '0') then 
                        fwdata(35 downto 32)<= MERGER_FIFO_RUN_END_MARKER;
                        fwrite              <= '1';
                        run_shutdown        <= '0';
                    else
                        mischief_managed    <= '1';
                    end if;
                    genstate                <= idle;

                when others =>
                    genstate <= idle;
            end case;

        end if;
    end process;

    shift0 : entity work.linear_shift
    generic map(
        g_m     => 65,
        g_poly  => lfsr_taps65
    )
    port map(
        i_clk               => i_clk,
        reset_n             => '1',
        i_sync_reset        => reset,
        i_seed              => "11001111101100010101110100100010011010110001101011110100101010010",
        i_en                => '1',
        o_lfsr              => random0--,
    );

    tot         <= random0( 5 downto  0);
    col         <= random0(13 downto  6);
    row         <= random0(21 downto 14);
    chipID_index<= random0(27 downto 22);
    chipID      <= valid_chipIDs(to_integer(unsigned(chipID_index)) mod valid_chipIDs'length);

end architecture;