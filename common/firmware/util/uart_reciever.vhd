LIBRARY IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all; --required for arithmetic with std_logic

-- UART RECIEVER
-- recieves (serial) 9 bit words with a predefined baud rate
-- returns 9 bit words synchrones with the system clock
-- idle level is 1
-- start bit is 0
-- stop bit is 1

entity uart_reciever is
generic (
    Clk_Ratio : integer := 100
);
port (
    Clk         : in    std_logic;
    Reset       : in    std_logic; -- here asynchronous reset
    DataIn      : in    std_logic; -- enable
    ReadEnable  : out   std_logic; -- down when 1, else up
    DataOut     : out   std_logic_vector(8 downto 0)
);
end entity;

architecture rtl of uart_reciever is

    signal Reset_Counter : std_logic;
    signal CounterOut : unsigned(31 downto 0);

    type fsm is (Idle, S_StartBit, Reading, EndState);
    signal present_state : fsm;

    signal wait_period : Integer;
    signal wait_period_init : Integer;

    signal DataBit : unsigned(3 downto 0); -- which bit of the out word is to be set
    signal DataCache : unsigned(8 downto 0); -- 9 bits for mscb protocol, bit 9 is the addressing bit

    signal SetReadEnable : std_logic;
    signal extra : std_logic;

begin

    -- wire up components
    counter_i : entity work.counter_async
    port map (
        Clk         => Clk,
        Reset       => Reset_Counter,
        Enable      => '1',
        CountDown   => '0',
        CounterOut  => CounterOut,
        Init        => to_unsigned(0,32)
    );

    wait_period <= Clk_Ratio-1-2;
    wait_period_init <= (Clk_Ratio-1)/2-2;


    -- main process
    process(Clk, Reset)
    begin
    if Reset = '1' then
      ReadEnable <= '0';
      SetReadEnable <= '0';
      DataOut <= (others => '0');
      DataCache <= (others => '0');
      present_state <= Idle;
      Reset_Counter <= '1';
      DataBit <= "0000";

    elsif rising_edge(Clk) then

        if SetReadEnable = '0' then -- make read enable when ready
            ReadEnable <= '0';
        end if;

        case present_state is
        when Idle =>
            Reset_Counter <= '1';
            DataBit <= "0000";
            if DataIn = '0' then -- in the mscb protocol, idle is '1', and '0' is the start bit
                present_state <= S_StartBit; -- start bit found, state, transition, counter starts
                Reset_Counter <= '0';
            end if;

        when S_StartBit =>
            if CounterOut > to_unsigned(wait_period_init,16) then
                present_state <= Reading;
                Reset_Counter <= '1'; -- reset clock once we waited 0.5 clock cycles
            end if; -- wait 1.5 data in clock cycles

        when Reading =>
            if  (CounterOut > to_unsigned(wait_period,16) ) then
                Reset_Counter <= '1';
                DataCache(to_integer(DataBit)) <= DataIn; -- Read a bit, and shift the read bit by 1
                DataBit <= DataBit + 1;
            else
                Reset_Counter <= '0'; -- start the counter
            end if;

            if DataBit = "1001" then -- if all bits are filled, DataCache is full
                present_state <= EndState;
                Reset_Counter <= '1';
            end if;

        when EndState =>
            if CounterOut > to_unsigned(wait_period,16) then
                if DataIn = '1' then -- check stop bit
                    DataOut <= std_logic_vector(DataCache);
                    SetReadEnable <= '1';
                end if;
            end if;
            if CounterOut > 2*to_unsigned(wait_period,16) then
                present_state <= Idle;
                if SetReadEnable = '1' then
                    ReadEnable <= '1';
                    SetReadEnable <='0';
                end if;
            else
                Reset_Counter <= '0';
            end if;

        end case; -- end state machine checks


    end if; -- end rising edge trigger
    end process;

end architecture;
