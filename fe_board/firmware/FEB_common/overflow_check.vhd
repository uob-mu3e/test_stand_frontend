-- overflow check for the merger fifos
-- writing into an almost full merger fifo will corrupt the packet
-- this entity throws the packet away and sends an error instead of corrupting it

-- Martin Mueller, Jan 2020

library ieee;
use ieee.std_logic_1164.all;

entity overflow_check is
--generic (
--);
port (
    i_clk       : in  std_logic;
    i_reset     : in  std_logic;
    i_write_req : in  std_logic;
    i_wdata     : in  std_logic_vector(35 downto 0);
    o_write_req : out std_logic;
    o_wdata     : out std_logic_vector(35 downto 0)--;
);
end entity;

architecture rtl of overflow_check is

begin

-- dummy
    o_write_req <= i_write_req;
    o_wdata     <= i_wdata;

--    process(i_clk, i_reset)
--    begin
--        if ( i_reset = '1' ) then
--            
--        elsif rising_edge(i_clk) then
--            
--        end if;
--    end process;

end architecture;