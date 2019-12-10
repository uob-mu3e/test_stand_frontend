--
-- single clock fifo
--
-- author : Alexandr Kozlinskiy
-- date : 2019-08-07
--

library ieee;
use ieee.std_logic_1164.all;

--
-- single clock fallthrough fifo
--
entity fifo_sc is
generic (
    DATA_WIDTH_g : positive := 8;
    ADDR_WIDTH_g : positive := 8--;
);
port (
    o_rempty    : out   std_logic;
    i_re        : in    std_logic;
    o_rdata     : out   std_logic_vector(DATA_WIDTH_g-1 downto 0);

    o_wfull     : out   std_logic;
    i_we        : in    std_logic;
    i_wdata     : in    std_logic_vector(DATA_WIDTH_g-1 downto 0);

    i_reset_n   : in    std_logic;
    i_clk       : in    std_logic--;
);
end entity;

library ieee;
use ieee.numeric_std.all;

architecture arch of fifo_sc is

    type ram_t is array (2**ADDR_WIDTH_g-1 downto 0) of std_logic_vector(DATA_WIDTH_g-1 downto 0);
    signal ram : ram_t;

    subtype addr_t is unsigned(ADDR_WIDTH_g-1 downto 0);
    subtype ptr_t is unsigned(ADDR_WIDTH_g downto 0);

    constant XOR_FULL_c : ptr_t := "10" & ( ADDR_WIDTH_g-2 downto 0 => '0' );

    signal re, we : std_logic;
    signal rempty, wfull : std_logic;
    signal rptr, wptr : ptr_t := (others => '0');

begin

    o_rempty <= rempty;
    o_wfull <= wfull;

    -- check for underflow and overflow
    re <= ( i_re and not rempty );
    we <= ( i_we and not wfull );

    process(i_clk, i_reset_n)
        variable rptr_v, wptr_v : ptr_t;
    begin
    if ( i_reset_n = '0' ) then
        rempty <= '1';
        wfull <= '1';
        rptr <= (others => '0');
        wptr <= (others => '0');
        --
    elsif rising_edge(i_clk) then
        if ( we = '1' ) then
            ram(to_integer(wptr(addr_t'range))) <= i_wdata;
        end if;

        -- advance pointers
        rptr_v := rptr + ("" & re);
        wptr_v := wptr + ("" & we);
        rptr <= rptr_v;
        wptr <= wptr_v;

        rempty <= work.util.to_std_logic( rptr_v = wptr_v );
        wfull <= work.util.to_std_logic( (rptr_v xor wptr_v) = XOR_FULL_c );

        --
    end if; -- rising_edge
    end process;

    o_rdata <= ram(to_integer(rptr(addr_t'range)));

end architecture;
