--

library ieee;
use ieee.std_logic_1164.all;

entity ip_scfifo is
generic (
    ADDR_WIDTH          : positive := 8;
    DATA_WIDTH          : positive := 8;
    SHOWAHEAD           : string   := "ON";
    DEVICE              : string   := "Arria 10"--;
);
port (
    clock           : in    std_logic;
    data            : in    std_logic_vector(DATA_WIDTH-1 downto 0);
    rdreq           : in    std_logic;
    sclr            : in    std_logic;
    wrreq           : in    std_logic;
    almost_empty    : out   std_logic;
    almost_full     : out   std_logic;
    empty           : out   std_logic;
    full            : out   std_logic;
    q               : out   std_logic_vector(DATA_WIDTH-1 downto 0);
    usedw           : out   std_logic_vector(ADDR_WIDTH-1 downto 0)
);
end entity;

architecture arch of ip_scfifo is

begin

    e_ip_scfifo_v2 : entity work.ip_scfifo_v2
    generic map (
        g_ADDR_WIDTH => ADDR_WIDTH,
        g_DATA_WIDTH => DATA_WIDTH,
        g_SHOWAHEAD => SHOWAHEAD,
        g_WREG_N => 0,
        g_RREG_N => 0,
        g_DEVICE_FAMILY => DEVICE--,
    )
    port map (
        o_usedw         => usedw,

        i_wdata         => data,
        i_we            => wrreq,
        o_wfull         => full,
        o_almost_full   => almost_full,

        o_rdata         => q,
        i_rack          => rdreq,
        o_rempty        => empty,
        o_almost_empty  => almost_empty,

        i_clk           => clock,
        i_reset_n       => not sclr--,
    );

end architecture;
