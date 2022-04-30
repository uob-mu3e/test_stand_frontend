library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


-- merge packets delimited by SOP and EOP from N input streams
entity swb_stream_merger is
generic (
    g_ADDR_WIDTH : positive := 8;
    g_set_type : boolean := false;
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    work.mu3e.link_array_t(N-1 downto 0);
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    i_rmask_n   : in    std_logic_vector(N-1 downto 0);
    i_en        : in    std_logic;
    o_rack      : out   std_logic_vector(N-1 downto 0);

    -- output stream
    o_wdata     : out   work.mu3e.link_t;
    o_rempty    : out   std_logic;
    i_ren       : in    std_logic;

    -- output stream debug
    o_wdata_debug     : out   work.mu3e.link_t;
    o_rempty_debug    : out   std_logic;
    i_ren_debug       : in    std_logic;

    --! status counters
    --! 0: farm fifo full
    --! 1: debug fifo almost full
    o_counters  : out work.util.slv32_array_t(1 downto 0);

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of swb_stream_merger is

    -- data path farm signals
    signal rempty : std_logic_vector(N-1 downto 0);
    signal wdata, rdata : work.mu3e.link_t;
    signal wfull, we : std_logic;

    -- debug path signals
    type write_debug_type is (idle, write_data, skip_data);
    signal write_debug_state : write_debug_type;
    signal wrusedw : std_logic_vector(g_ADDR_WIDTH - 1 downto 0);
    signal wdata_debug : work.mu3e.link_t;
    signal almost_full, we_debug : std_logic;

begin

    --! counters
    e_cnt_e_stream_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(0), i_ena => wfull, i_reset_n => i_reset_n, i_clk => i_clk );

    e_cnt_e_debug_stream_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(1), i_ena => almost_full, i_reset_n => i_reset_n, i_clk => i_clk );

    rempty <= i_rempty or not i_rmask_n when i_en = '1' else (others => '1');

    e_stream_merger : entity work.stream_merger
    generic map (
        g_set_type => g_set_type,
        N => N--,
    )
    port map (
        -- input stream
        i_rdata     => i_rdata,
        i_rempty    => rempty,
        o_rack      => o_rack,

        -- output stream
        o_wdata     => wdata,
        i_wfull     => wfull,
        o_we        => we,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    e_stream_fifo_farm : entity work.link_scfifo
    generic map (
        g_ADDR_WIDTH    => g_ADDR_WIDTH,
        g_RREG_N        => 1--, -- TNS=-900
    )
    port map (
        i_wdata         => wdata,
        i_we            => we,
        o_wfull         => wfull,

        o_rdata         => rdata,
        o_rempty        => o_rempty,
        i_rack          => i_ren,

        i_clk           => i_clk,
        i_reset_n       => i_reset_n--,
    );
    o_wdata <= rdata;

    --! write data to debug fifo
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        wdata_debug         <= work.mu3e.LINK_ZERO;
        we_debug            <= '0';
        write_debug_state   <= idle;
        --
    elsif ( rising_edge(i_clk) ) then

        wdata_debug <= rdata;
        we_debug    <= '0';

        if ( i_ren = '0' ) then
            --
        else
            case write_debug_state is
            when idle =>
                -- start on start of package
                if ( rdata.sop = '1' ) then
                    if ( almost_full = '1' ) then
                        write_debug_state   <= skip_data;
                    else
                        write_debug_state   <= write_data;
                        we_debug            <= '1';
                    end if;
                end if;

            when write_data =>
                we_debug            <= '1';
                -- stop on end of package
                if ( rdata.eop = '1' ) then
                    write_debug_state   <= idle;
                end if;

            when skip_data =>
                -- stop on end of package
                if ( rdata.eop = '1' ) then
                    write_debug_state <= idle;
                end if;

            when others =>
                write_debug_state <= idle;

            end case;
        end if;
        --
    end if;
    end process;

    e_stream_fifo_debug : entity work.link_scfifo
    generic map (
        g_ADDR_WIDTH    => g_ADDR_WIDTH,
        g_RREG_N        => 1--, -- TNS=-900
    )
    port map (
        i_wdata         => wdata_debug,
        i_we            => we_debug,
        o_wfull         => open, -- we dont use the full since we check wrusedw
        o_usedw         => wrusedw,

        o_rdata         => o_wdata_debug,
        o_rempty        => o_rempty_debug,
        i_rack          => i_ren_debug,

        i_clk           => i_clk,
        i_reset_n       => i_reset_n--,
    );

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        almost_full <= '0';
    elsif ( rising_edge(i_clk) ) then
        if ( wrusedw(g_ADDR_WIDTH - 1) = '1' ) then
            almost_full <= '1';
        else
            almost_full <= '0';
        end if;
    end if;
    end process;

end architecture;
