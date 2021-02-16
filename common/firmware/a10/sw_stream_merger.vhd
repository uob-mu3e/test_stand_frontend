library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.dataflow_components.all;

-- merge packets delimited by SOP and EOP from N input streams
entity sw_stream_merger is
generic (
    W : positive := 32;
    N : positive--;
);
port (
    -- input streams
    i_rdata     : in    data_array(N - 1 downto 0);
    i_rsop      : in    std_logic_vector(N-1 downto 0); -- start of packet (SOP)
    i_reop      : in    std_logic_vector(N-1 downto 0); -- end of packet (EOP)
    i_rempty    : in    std_logic_vector(N-1 downto 0);
    o_rack      : out   std_logic_vector(N-1 downto 0); -- read ACK

    -- output stream
    o_wdata     : out   std_logic_vector(W-1 downto 0);
    o_wsop      : out   std_logic; -- SOP
    o_weop      : out   std_logic; -- EOP
    i_wfull     : in    std_logic;
    o_we        : out   std_logic; -- write enable

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

architecture arch of sw_stream_merger is

    signal rdata : std_logic_vector(N*W-1 downto 0);

begin

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
