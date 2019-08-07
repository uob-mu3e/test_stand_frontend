--
-- single clock fifo
--
-- author : Alexandr Kozlinskiy
-- date : 2019-08-07
--

library ieee;
use ieee.std_logic_1164.all;

entity scfifo is
generic (
    DATA_WIDTH_g : positive := 8;
    ADDR_WIDTH_g : positive := 8--;
);
port (
    o_wfull     : out   std_logic;
    i_we        : in    std_logic;
    i_wdata     : in    std_logic_vector(DATA_WIDTH_g-1 downto 0);

    o_rempty    : out   std_logic;
    o_rdata     : out   std_logic_vector(DATA_WIDTH_g-1 downto 0);
    i_re        : in    std_logic;

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

library ieee;
use ieee.numeric_std.all;

architecture arch of scfifo is

    type ram_t is array (2**ADDR_WIDTH_g-1 downto 0) of std_logic_vector(DATA_WIDTH_g-1 downto 0);
    signal ram : ram_t;

    subtype addr_t is unsigned(ADDR_WIDTH_g-1 downto 0);
    subtype ptr_t is unsigned(ADDR_WIDTH_g downto 0);

    constant XOR_FULL_c : ptr_t := "10" & ( ADDR_WIDTH_g-2 downto 0 => '0' );

    signal we, re : std_logic;
    signal wfull, rempty : std_logic;
    signal wptr, rptr : ptr_t := (others => '0');

begin

    o_wfull <= wfull;
    o_rempty <= rempty;
    we <= ( i_we and not wfull );
    re <= ( i_re and not rempty );

    process(i_clk, i_reset_n)
        variable wptr_v, rptr_v : ptr_t;
    begin
    if ( i_reset_n = '0' ) then
        wfull <= '1';
        rempty <= '1';
        wptr <= (others => '0');
        rptr <= (others => '0');
        --
    elsif rising_edge(i_clk) then
        if ( we = '1' ) then
            ram(to_integer(wptr(addr_t'range))) <= i_wdata;
        end if;

        wptr_v := wptr + ("" & we);
        rptr_v := rptr + ("" & re);
        wptr <= wptr_v;
        rptr <= rptr_v;

        wfull <= work.util.to_std_logic( (rptr_v xor wptr_v) = XOR_FULL_c );
        rempty <= work.util.to_std_logic( rptr_v = wptr_v );

        --
    end if; -- rising_edge
    end process;

    o_rdata <= ram(to_integer(rptr(addr_t'range)));

end architecture;
