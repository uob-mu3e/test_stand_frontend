library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


-- merge packets delimited by SOP and EOP from N input streams
entity swb_stream_merger is
generic (
    W : positive := 34;
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    work.util.slv34_array_t(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0);
    i_reop      : in    std_logic_vector(N-1 downto 0);
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    i_rmask_n   : in    std_logic_vector(N-1 downto 0);
    i_en        : in    std_logic;
    o_rack      : out   std_logic_vector(N-1 downto 0);

    -- output stream
    o_wdata     : out   std_logic_vector(31 downto 0);
    o_rempty    : out   std_logic;
    i_ren       : in    std_logic;
    o_wsop      : out   std_logic;
    o_weop      : out   std_logic;

    --! status counters 
    --! 0: e_stream_fifo full
    o_counters  : out work.util.slv32_array_t(0 downto 0);

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of swb_stream_merger is

    signal rdata : std_logic_vector(N*W-1 downto 0);
    signal rempty : std_logic_vector(N-1 downto 0);
    signal wdata, q_stream : std_logic_vector(33 downto 0);
    signal datak : std_logic_vector(3 downto 0);
    signal wfull, we : std_logic;

begin

    --! counters
    e_cnt_e_stream_fifo_full : entity work.counter
    generic map ( WRAP => true, W => 32 )
    port map ( o_cnt => o_counters(0), i_ena => wfull, i_reset_n => i_reset_n, i_clk => i_clk );

    --! map data for stream merger
    generate_rdata : for i in 0 to N-1 generate
        rdata(W-1 + i*W downto i*W) <= i_rdata(i);
    end generate;

    rempty <=   i_rempty or not i_rmask_n when i_en = '1' else
                (others => '1');
    e_stream_merger : entity work.stream_merger
    generic map (
        W => W,
        N => N--,
    )
    port map (
        i_rdata     => rdata,
        i_rsop      => i_rsop,
        i_reop      => i_reop,
        i_rempty    => rempty,
        o_rack      => o_rack,

        -- output stream
        o_wdata(33 downto 0) => wdata,
        o_wsop      => open,
        o_weop      => open,
        i_wfull     => wfull,
        o_we        => we,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    e_stream_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH  => 8,
        DATA_WIDTH  => 34,
        REGOUT      => 0,
        RAM_OUT_REG => "ON",
        DEVICE      => "Arria 10"--,
    )
    port map (
        q               => q_stream,
        empty           => o_rempty,
        rdreq           => i_ren,
        data            => wdata,
        full            => wfull,
        wrreq           => we,
        sclr            => not i_reset_n,
        clock           => i_clk--,
    );

    --! only output data not datak
    o_wdata <= q_stream(31 downto 0);
    o_wsop <= '1' when q_stream(33 downto 32) = "10" else '0';
    o_weop <= '1' when q_stream(33 downto 32) = "01" else '0';

end architecture;
