-- overflow check for the merger fifos
-- writing into an almost full merger fifo will corrupt the packet
-- this entity throws the packet away and sends an error instead of corrupting it

-- Martin Mueller, Jan 2020

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity overflow_check is
generic (
    FIFO_ADDR_WIDTH     : positive := 10;
    MAX_PAKET_SIZE      : positive := 100--;
);
port (
    i_clk               : in  std_logic;
    i_reset             : in  std_logic;
    i_write_req         : in  std_logic;
    i_wdata             : in  std_logic_vector(35 downto 0);
    i_usedw             : in  std_logic_vector(FIFO_ADDR_WIDTH-1 downto 0 );
    o_write_req         : out std_logic;
    o_wdata             : out std_logic_vector(35 downto 0)--;
);
end entity;

architecture rtl of overflow_check is

begin

    process(i_clk, i_reset)
    begin
        if ( i_reset = '1' ) then
            o_write_req         <= '0';
            o_wdata             <= (others => '0');
        elsif rising_edge(i_clk) then
            if (unsigned(i_usedw) < (2**FIFO_ADDR_WIDTH)-MAX_PAKET_SIZE) then
                o_write_req     <= i_write_req;
                o_wdata         <= i_wdata;
            end if;
        end if;
    end process;

end architecture;