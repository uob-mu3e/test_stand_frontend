library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;


entity swb_time_merger is
generic (
    g_ADDR_WIDTH : positive := 11;
    g_NLINKS_DATA : positive := 8;
    -- Data type: x"00" = pixel, x"01" = scifi, "10" = tiles
    DATA_TYPE : std_logic_vector(1 downto 0) := "00"--;
);
port (
    -- input streams
    i_rx            : in    work.mu3e.link_array_t(g_NLINKS_DATA-1 downto 0);
    i_rempty        : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0) := (others => '1');
    i_rmask_n       : in    std_logic_vector(g_NLINKS_DATA - 1 downto 0);
    o_rack          : out   std_logic_vector(g_NLINKS_DATA - 1 downto 0);

    -- TODO: add me
    o_counters      : out   work.util.slv32_array_t(6 downto 0);

    -- output stream
    o_wdata         : out   work.mu3e.link_t;
    o_rempty        : out   std_logic;
    i_ren           : in    std_logic;
    
    -- output stream debug
    o_wdata_debug   : out   work.mu3e.link_t;
    o_rempty_debug  : out   std_logic;
    i_ren_debug     : in    std_logic;
    
    o_error         : out   std_logic;

    i_en            : in    std_logic;
    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of swb_time_merger is

    -- data path farm signals
    signal rdata : work.mu3e.link_t;

    -- debug path signals
    type write_debug_type is (idle, write_data, skip_data);
    signal write_debug_state : write_debug_type;
    signal wrusedw : std_logic_vector(g_ADDR_WIDTH - 1 downto 0);
    signal wdata_debug, q_stream_debug : work.mu3e.link_t;
    signal almost_full, we_debug : std_logic;

begin

    e_time_merger : entity work.time_merger
    generic map (
        g_ADDR_WIDTH => g_ADDR_WIDTH,
        g_NLINKS_DATA => g_NLINKS_DATA,
        DATA_TYPE => DATA_TYPE--,
    )
    port map (
        -- input streams
        i_data                  => i_rx,
        i_empty                 => i_rempty,
        i_mask_n                => i_rmask_n,
        o_rack                  => o_rack,

        -- output stream
        --! TODO: cnt errors, at the moment they are sent out at the end of normal event
        o_rdata                 => rdata,
        i_rack                  => i_ren,
        o_empty                 => o_rempty,

        i_en                    => i_en,
        i_reset_n               => i_reset_n,
        i_clk                   => i_clk--,
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
