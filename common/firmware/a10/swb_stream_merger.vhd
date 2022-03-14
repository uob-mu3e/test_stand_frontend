library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


-- merge packets delimited by SOP and EOP from N input streams
entity swb_stream_merger is
generic (
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

    --! status counters
    --! 0: e_stream_fifo full
    o_counters  : out work.util.slv32_array_t(0 downto 0);

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of swb_stream_merger is

    signal rdata : std_logic_vector(work.mu3e.LINK_LENGTH*N-1 downto 0);
    signal rsop : std_logic_vector(N-1 downto 0);
    signal reop : std_logic_vector(N-1 downto 0);
    signal rempty : std_logic_vector(N-1 downto 0);

    signal wdata : std_logic_vector(work.mu3e.LINK_LENGTH-1 downto 0);
    signal wfull, we : std_logic;

begin

    --! counters
    e_cnt_e_stream_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(0), i_ena => wfull, i_reset_n => i_reset_n, i_clk => i_clk );

    --! map data for stream merger
    generate_rdata : for i in 0 to N-1 generate
        rdata(work.mu3e.LINK_LENGTH*i + work.mu3e.LINK_LENGTH-1 downto 0 + work.mu3e.LINK_LENGTH*i) <= work.mu3e.to_slv(i_rdata(i));
        rsop(i) <= i_rdata(i).sop;
        reop(i) <= i_rdata(i).eop;
    end generate;

    rempty <=   i_rempty or not i_rmask_n when i_en = '1' else
                (others => '1');
    e_stream_merger : entity work.stream_merger
    generic map (
        W => work.mu3e.LINK_LENGTH,
        N => N--,
    )
    port map (
        o_rack      => o_rack,
        i_rdata     => rdata,
        i_rsop      => rsop,
        i_reop      => reop,
        i_rempty    => rempty,

        -- output stream
        o_we        => we,
        o_wdata     => wdata,
        o_wsop      => open,
        o_weop      => open,
        i_wfull     => wfull,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    e_stream_fifo : entity work.link_scfifo
    generic map (
        g_ADDR_WIDTH => 8,
        g_RREG_N => 1--,
    )
    port map (
        i_we            => we,
        i_wdata         => work.mu3e.to_link(wdata),
        o_wfull         => wfull,

        i_rack          => i_ren,
        o_rdata         => o_wdata,
        o_rempty        => o_rempty,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );

end architecture;
