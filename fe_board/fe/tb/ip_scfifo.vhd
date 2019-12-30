library ieee;
use ieee.std_logic_1164.all;

entity ip_scfifo is
generic (
    ADDR_WIDTH : positive := 8;
    DATA_WIDTH : positive := 8--;
);
port (
    full            : out   std_logic;
    almost_full     : out   std_logic;
    wrreq           : in    std_logic;
    data            : in    std_logic_vector(DATA_WIDTH-1 downto 0);

    empty           : out   std_logic;
    almost_empty    : out   std_logic;
    q               : out   std_logic_vector(DATA_WIDTH-1 downto 0);
    rdreq           : in    std_logic;

    usedw           : out   std_logic_vector(ADDR_WIDTH-1 downto 0);

    sclr            : in    std_logic;
    clock           : in    std_logic--;
);
end entity;

architecture arch of ip_scfifo is

    signal reset_n : std_logic;

begin

    reset_n <= not sclr;

    e_fifo : entity work.fifo_sc
    generic map (
        DATA_WIDTH_g => DATA_WIDTH,
        ADDR_WIDTH_g => ADDR_WIDTH--,
    )
    port map (
        i_wdata     => data,
        i_we        => wrreq,
        o_wfull     => full,

        o_rdata     => q,
        i_rack      => rdreq,
        o_rempty    => empty,

        i_reset_n   => reset_n,
        i_clk       => clock--;
    );

    almost_full <= 'X';
    almost_empty <= 'X';
    usedw <= (others => 'X');

end architecture;
