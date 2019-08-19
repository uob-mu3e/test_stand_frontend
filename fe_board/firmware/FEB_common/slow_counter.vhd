LIBRARY IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all; --required for arithmetic with std_logic

-- up/down counter, input clock is scaled down
entity slow_counter is
generic (
    Clk_Ratio : integer := 100
);
port (
    Clk         : in    std_logic;
    Reset       : in    std_logic; -- here asynchronous reset
    Enable      : in    std_logic; -- enable
    CountDown   : in    std_logic; -- down when 1, else up
    CounterOut  : out   unsigned(15 downto 0);
    Init        : in    unsigned(15 downto 0)
);
end entity;

architecture rtl of slow_counter is

    signal cnt : unsigned(15 downto 0);
    signal scale_down_counter : unsigned(15 downto 0);
    signal slow_clock : std_logic; 

begin

    process(Clk, Reset)
    begin
    if(Reset = '1') then
        slow_clock <= '0';
        scale_down_counter <= (others => '0');
    elsif rising_edge(Clk) then
        if (scale_down_counter = Clk_Ratio) then
            slow_clock <= not slow_clock;
            scale_down_counter <= (others => '0');
        else
            scale_down_counter <= scale_down_counter + 1;
        end if;
    end if;
    end process;



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

        end if; -- end 'enable'
    end if; -- end rising_edge
    end process;

    CounterOut <= cnt;

end architecture;
