--
-- fifo read register
--
-- author : Alexandr Kozlinskiy
-- date : 2021-06-11
--

library ieee;
use ieee.std_logic_1164.all;

--
-- @deprecated use fifo_reg
--
entity fifo_rreg is
generic (
    g_DATA_WIDTH : positive := 32;
    g_N : natural := 2--;
);
port (
    o_rdata         : out   std_logic_vector(g_DATA_WIDTH-1 downto 0);
    i_re            : in    std_logic;
    o_rempty        : out   std_logic;

    i_fifo_rdata    : in    std_logic_vector(g_DATA_WIDTH-1 downto 0);
    o_fifo_re       : out   std_logic;
    i_fifo_rempty   : in    std_logic;

    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of fifo_rreg is

    signal fifo_re_n : std_logic;

begin

    o_fifo_re <= not fifo_re_n;

    e_fifo_reg : entity work.fifo_reg
    generic map (
        g_DATA_WIDTH => g_DATA_WIDTH,
        g_N => g_N--,
    )
    port map (
        o_rdata     => o_rdata,
        o_rempty    => o_rempty,
        i_rack      => i_re,

        i_wdata     => i_fifo_rdata,
        i_we        => not i_fifo_rempty,
        o_wfull     => fifo_re_n,

        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

end architecture;
