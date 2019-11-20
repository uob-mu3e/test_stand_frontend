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

    signal wfull, rempty, reset : std_logic;

begin

    e_fifo : entity work.scfifo
    generic map (
        DATA_WIDTH_g => DATA_WIDTH,
        ADDR_WIDTH_g => ADDR_WIDTH--,
    )
    port map (
        o_wfull     => wfull,
        i_we        => wrreq,
        i_wdata     => data,

        o_rempty    => rempty,
        o_rdata     => q,
        i_re        => rdreq,

        i_reset_n   => reset,
        i_clk       => clock--;
    );

    full <= wfull;
    almost_full <= 'X';
    empty <= rempty;
    almost_empty <= 'X';

    reset <= not sclr;

    usedw <= (others => 'X');

end architecture;
