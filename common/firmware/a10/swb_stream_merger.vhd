library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.dataflow_components.all;

-- merge packets delimited by SOP and EOP from N input streams
entity swb_stream_merger is
generic (
    W : positive := 32;
    N : positive--;
);
port (
    -- input streams
    i_rx        : in    data_array(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0);
    i_reop      : in    std_logic_vector(N-1 downto 0);
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    i_rmask_n   : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0);

    -- output stream
    o_q         : out   std_logic_vector(31 downto 0);
    o_rempty    : out   std_logic;
    i_ren       : in    std_logic;
    o_header    : out   std_logic;
    o_trailer   : out   std_logic;

    --! status counters 
    --! 0: e_stream_fifo full
    o_counters  : out work.util.slv32_array_t(0 downto 0);

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of swb_stream_merger is

    signal rdata : std_logic_vector(N*W-1 downto 0);
    signal wdata, rdata : std_logic_vector(35 downto 0);
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

    e_stream_merger : entity work.stream_merger
    generic map (
        W => W,
        N => N--,
    )
    port map (
        i_rdata     => rdata,
        i_rsop      => i_rsop,
        i_reop      => i_reop,
        i_rempty    => i_rempty or not i_rmask_n,
        o_rack      => o_rack,

        -- output stream
        o_wdata(35 downto 0) => wdata,
        o_wsop      => open,
        o_weop      => open,
        i_wfull     => wfull,
        o_we        => we,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    e_stream_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 36,
        DEVICE => "Arria 10"--,
    )
    port map (
        q               => rdata,
        empty           => o_rempty,
        rdreq           => i_ren,
        data            => wdata,
        full            => wfull,
        wrreq           => we,
        sclr            => not i_reset_n,
        clock           => i_clk--,
    );

    --! only output data not datak
    o_q <= rdata(35 downto 4);
    datak <= rdata(3 downto 0);
    
    o_header <=
        '1' when datak = "0001" and o_q(7 downto 0) = x"BC"
        else '0';
    o_trailer <=
        '1' when datak = "0001" and o_q(7 downto 0) = x"9C"
        else '0';

end architecture;
