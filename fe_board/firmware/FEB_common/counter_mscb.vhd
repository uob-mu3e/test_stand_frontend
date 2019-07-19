LIBRARY IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all; --required for arithmetic with std_logic

-- up/down counter
entity counter_mscb is
port(
     Clk	: in  std_logic;
     Reset		: in  std_logic; -- here asynchronous reset
     Enable		: in  std_logic; -- enable
     CountDown	: in  std_logic; -- down when 1, else up
     CounterOut	: out unsigned(31 downto 0);
     Init : in unsigned(31 downto 0)
);
end counter_mscb;

architecture a of counter_mscb is

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

      end if; -- end 'enable'

    end if; -- end rising edge

  end process;

  CounterOut <= (cnt);

end;
