library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- merge packets delimited by SOP and EOP from N input streams
entity swb_stream_merger is
generic (
    g_ADDR_WIDTH : positive := 8;
    W : positive := 32;
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    work.util.slv32_array_t(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0);
    i_reop      : in    std_logic_vector(N-1 downto 0);
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    i_rmask_n   : in    std_logic_vector(N-1 downto 0);
    i_en        : in    std_logic;
    o_rack      : out   std_logic_vector(N-1 downto 0);

    -- output stream
    o_wdata     : out   std_logic_vector(W-1 downto 0);
    o_rempty    : out   std_logic;
    i_ren       : in    std_logic;
    o_wsop      : out   std_logic;
    o_weop      : out   std_logic;

    -- output stream debug
    o_wdata_debug     : out   std_logic_vector(W-1 downto 0);
    o_rempty_debug    : out   std_logic;
    i_ren_debug       : in    std_logic;
    o_wsop_debug      : out   std_logic;
    o_weop_debug      : out   std_logic;

    --! status counters
    --! 0: e_stream_fifo full
    o_counters  : out work.util.slv32_array_t(0 downto 0);

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of swb_stream_merger is

    -- data path farm signals
    signal rdata : std_logic_vector(N*W-1 downto 0);
    signal rempty : std_logic_vector(N-1 downto 0);
    signal wdata : std_logic_vector(W-1 downto 0);
    signal q_stream, wdata_farm : std_logic_vector(2+W-1 downto 0);
    signal wfull, we, wsop, weop : std_logic;

    -- debug path signals
    type write_debug_type is (idle, write_data, skip_data);
    signal write_debug_state : write_debug_type;
    signal wrusedw : std_logic_vector(g_ADDR_WIDTH - 1 downto 0);
    signal wdata_debug, q_stream_debug : std_logic_vector(2+W-1 downto 0);
    signal almost_full, we_debug : std_logic;

begin

    --! counters
    e_cnt_e_stream_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(0), i_ena => wfull, i_reset_n => i_reset_n, i_clk => i_clk );

    --! map data for stream merger
    generate_rdata : for i in 0 to N-1 generate
        rdata(W-1 + i*W downto i*W) <= i_rdata(i);
    end generate;

    rempty <= i_rempty or not i_rmask_n when i_en = '1' else (others => '1');

    e_stream_merger : entity work.stream_merger
    generic map (
        W => W,
        N => N--,
    )
    port map (
        -- input stream
        i_rdata     => rdata,
        i_rsop      => i_rsop,
        i_reop      => i_reop,
        i_rempty    => rempty,
        o_rack      => o_rack,

        -- output stream
        o_wdata     => wdata,
        o_wsop      => wsop,
        o_weop      => weop,
        i_wfull     => wfull,
        o_we        => we,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    wdata_farm <= wsop & weop & wdata;
    e_stream_fifo_farm : entity work.ip_scfifo_v2
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH,
        g_DATA_WIDTH => 2+W,
        g_RREG_N => 1--, -- TNS=-900
    )
    port map (
        i_wdata         => wdata_farm,
        i_we            => we,
        o_wfull         => wfull,

        o_rdata         => q_stream,
        o_rempty        => o_rempty,
        i_rack          => i_ren,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    --! map output data
    o_wdata <= q_stream(W-1 downto 0);
    o_wsop  <= q_stream(2+W-1);
    o_weop  <= q_stream(1+W-1);

    --! write data to debug fifo
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        wdata_debug         <= (others => '0');
        we_debug            <= '0';
        write_debug_state   <= idle;
        --
    elsif rising_edge(i_clk) then

        wdata_debug <= q_stream;
        we_debug    <= '0';

        if ( i_ren = '0' ) then
            --
        else
            case write_debug_state is
            when idle =>
                -- start on start of package
                if ( q_stream(2+W-1) = '1' ) then
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
                if ( q_stream(1+W-1) = '1' ) then
                    write_debug_state   <= idle;
                end if;

            when skip_data =>
                -- stop on end of package
                if ( q_stream(1+W-1) = '1' ) then
                    write_debug_state <= idle;
                end if;

            when others =>
                write_debug_state <= idle;

            end case;
        end if;
        --
    end if;
    end process;

    e_stream_fifo_debug : entity work.ip_scfifo_v2
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH,
        g_DATA_WIDTH => wdata_debug'length,
        g_RREG_N => 1--,
    )
    port map (
        i_we            => we_debug,
        i_wdata         => wdata_debug,
        o_wfull         => open, -- we dont use the full since we check wrusedw
        o_usedw         => wrusedw,

        i_rack          => i_ren_debug,
        o_rdata         => q_stream_debug,
        o_rempty        => o_rempty_debug,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

    --! map output data debug
    o_wdata_debug   <= q_stream_debug(W-1 downto 0);
    o_wsop_debug    <= q_stream_debug(2+W-1);
    o_weop_debug    <= q_stream_debug(1+W-1);

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        almost_full <= '0';
    elsif rising_edge(i_clk) then
        if(wrusedw(g_ADDR_WIDTH - 1) = '1') then
            almost_full <= '1';
        else
            almost_full <= '0';
        end if;
    end if;
    end process;

end architecture;
