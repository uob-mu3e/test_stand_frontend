library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;


entity swb_time_merger is
generic (
    g_ADDR_WIDTH : positive := 11;
    g_NLINKS_DATA : positive := 8;
    -- Data type: x"01" = pixel, x"02" = scifi, x"03" = tiles
    DATA_TYPE: std_logic_vector(7 downto 0) := x"01"--;
);
port (
    -- input streams
    i_rx            : in    work.util.slv32_array_t(g_NLINKS_DATA - 1 downto 0);
    i_rsop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_reop          : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_rshop         : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_hit           : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_t0            : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_t1            : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    i_rempty        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0) := (others => '1');
    i_rmask_n       : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_rack          : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0);

    -- TODO: add me
    o_counters      : out   work.util.slv32_array_t(6 downto 0) := (others => '0');

    -- output stream
    o_wdata         : out   std_logic_vector(31 downto 0);
    o_rempty        : out   std_logic;
    i_ren           : in    std_logic;
    o_wsop          : out   std_logic;
    o_weop          : out   std_logic;

    -- output stream debug
    o_wdata_debug   : out   std_logic_vector(31 downto 0);
    o_rempty_debug  : out   std_logic;
    i_ren_debug     : in    std_logic;
    o_wsop_debug    : out   std_logic;
    o_weop_debug    : out   std_logic;

    o_error         : out   std_logic;

    i_en            : in    std_logic;
    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of swb_time_merger is

    -- data path farm signals
    signal wdata : std_logic_vector(31 downto 0);
    signal wsop, weop : std_logic;

    -- debug path signals
    type write_debug_type is (idle, write_data, skip_data);
    signal write_debug_state : write_debug_type;
    signal wrusedw : std_logic_vector(8 - 1 downto 0);
    signal wdata_debug, q_stream_debug : std_logic_vector(33 downto 0);
    signal almost_full, we_debug : std_logic;

begin

    e_time_merger : entity work.time_merger_v4
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH,
        g_NLINKS_DATA => g_NLINKS_DATA,
        DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input streams
        i_data                  => i_rx,
        i_sop                   => i_rsop,
        i_eop                   => i_reop,
        i_shop                  => i_rshop,
        i_hit                   => i_hit,
        i_t0                    => i_t0,
        i_t1                    => i_t1,
        i_empty                 => i_rempty,
        i_mask_n                => i_rmask_n,
        o_rack                  => o_rack,

        -- output stream
        o_wdata                 => wdata,
        o_wsop                  => wsop,
        o_weop                  => weop,
        i_rack                  => i_ren,
        o_empty                 => o_rempty,

        -- counters
        o_error                 => open,

        i_en                    => i_en,
        i_reset_n               => i_reset_n,
        i_clk                   => i_clk--,
    );

    --! map output data
    --! TODO: cnt errors, at the moment they are sent out at the end of normal event
    o_wdata <= wdata;
    o_wsop  <= wsop;
    o_weop  <= weop;

    --! write data to debug fifo
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n /= '1' ) then
        wdata_debug         <= (others => '0');
        we_debug            <= '0';
        write_debug_state   <= idle;
        --
    elsif ( rising_edge(i_clk) ) then

        wdata_debug <= wsop & weop & wdata;
        we_debug    <= '0';

        if ( i_ren = '0' ) then
            --
        else
            case write_debug_state is

            when idle =>
                -- start on start of package
                if ( wsop = '1' ) then
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
                if ( weop = '1' ) then
                    write_debug_state   <= idle;
                end if;

            when skip_data =>
                -- stop on end of package
                if ( weop = '1' ) then
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
        g_ADDR_WIDTH => 8,
        g_DATA_WIDTH => 34,
        g_RREG_N => 1--, -- TNS=-900
    )
    port map (
        i_wdata         => wdata_debug,
        i_we            => we_debug,
        o_wfull         => open, -- we dont use the full since we check wrusedw
        o_usedw         => wrusedw,

        o_rdata         => q_stream_debug,
        o_rempty        => o_rempty_debug,
        i_rack          => i_ren_debug,

        i_clk           => i_clk,
        i_reset_n       => i_reset_n--,
    );

    --! map output data debug
    o_wdata_debug   <= q_stream_debug(31 downto 0);
    o_wsop_debug    <= q_stream_debug(33);
    o_weop_debug    <= q_stream_debug(32);

    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        almost_full <= '0';
    elsif rising_edge(i_clk) then
        if(wrusedw(8 - 1) = '1') then
            almost_full <= '1';
        else
            almost_full <= '0';
        end if;
    end if;
    end process;

end architecture;
