-- counter with async reset

LIBRARY ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity counter_async is
port (
    Clk         : in    std_logic;
    Reset       : in    std_logic; 
    Enable      : in    std_logic; 
    CountDown   : in    std_logic; 
    CounterOut  : out   unsigned(31 downto 0);
    Init        : in    unsigned(31 downto 0)
);
end entity;

architecture a of counter_async is
    signal cnt : unsigned(31 downto 0);
begin

    process(Clk, Reset, Init)
    begin
    if Reset = '1' then
        cnt <= Init;
    elsif rising_edge(Clk) then
        if Enable = '1' then
            if( CountDown = '1' ) then
                cnt <= cnt - 1;
            else
                cnt <= cnt + 1;
            end if;
        end if;
    end if; 
    end process;
    CounterOut <= cnt;

end architecture;
