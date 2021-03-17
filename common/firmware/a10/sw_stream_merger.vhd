library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- merge packets delimited by SOP and EOP from N input streams
entity sw_stream_merger is
generic (
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    work.util.slv38_array_t(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0); -- start of packet (SOP)
    i_reop      : in    std_logic_vector(N-1 downto 0); -- end of packet (EOP)
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0); -- read ACK

    -- output stream
    o_wdata     : out   work.util.slv38_t;
    o_wsop      : out   std_logic; -- SOP
    o_weop      : out   std_logic; -- EOP
    i_wfull     : in    std_logic;
    o_we        : out   std_logic; -- write enable

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of sw_stream_merger is

    signal rdata : std_logic_vector(N*38-1 downto 0);

begin

    generate_rdata : for i in 0 to N-1 generate
        rdata((i+1)*38-1 downto i*38) <= i_rdata(i);
    end generate;

    e_stream_merger : entity work.stream_merger
    generic map (
        W => 38,
        N => N--,
    )
    port map (
        i_rdata     => rdata,
        i_rsop      => i_rsop,
        i_reop      => i_reop,
        i_rempty    => i_rempty,
        o_rack      => o_rack,

        -- output stream
        o_wdata     => o_wdata,
        o_wsop      => o_wsop,
        o_weop      => o_weop,
        i_wfull     => i_wfull,
        o_we        => o_we,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

end architecture;
