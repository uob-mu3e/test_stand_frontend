LIBRARY IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all; --required for arithmetic with std_logic

-- UART TRANSMITTER
-- recieves (parallel) 9 bit words 
-- returns 9 bit words with a predefined baud rate
-- idle level is 1
-- start bit is 0
-- stop bit is 1

entity uart_transmitter is
generic ( Clk_Ratio : integer := 100 );
port(
     Clk	: in  std_logic;
     Reset		: in  std_logic; -- here asynchronous reset
     DataIn		: in  std_logic_vector(8 downto 0);
     DataReady : in std_logic; -- data is ready to be read
     ReadRequest	: out  std_logic; 
     DataOut : out std_logic; 
	  Transmitting : out std_logic -- letting the outside world know data is being transmitted
);
end uart_transmitter;

architecture rtl of uart_transmitter is

component counter_mscb is
port(
     Clk	: in  std_logic;
     Reset		: in  std_logic; -- here asynchronous reset
     Enable		: in  std_logic; -- enable
     CountDown	: in  std_logic; -- down when 1, else up
     CounterOut	: out unsigned(31 downto 0);
	  Init : in unsigned(31 downto 0)
);
end component;

signal Reset_Counter : std_logic;
signal CounterOut : unsigned(31 downto 0);

type fsm is (Idle, S_StartBit, Sending, EndState);
signal present_state : fsm;

signal wait_period : Integer;
signal wait_period_init : Integer;

signal DataBit : unsigned(3 downto 0); -- which bit of the out word is to be set
signal SendReadRequest : std_logic;


begin

  -- wire up components
  counter_i : counter_mscb port map(
    Clk	=> Clk,
    Reset		=> Reset_Counter,
    Enable		=> '1',
    CountDown	=> '0',
    CounterOut => CounterOut,
    Init => to_unsigned(0,32)
  );

  -- the state machine does properly work with a waiting period of 2 or smaller between transitions
  wait_period <=  Clk_Ratio when Clk_Ratio > 3 else 3;
  
  
	 
	 
  -- main process
  process(Clk, Reset)
  begin

    if Reset = '1' then
      ReadRequest <= '0';
      DataOut <= '1';
      DataBit <= "0000";
      present_state <= Idle;
      SendReadRequest <= '0';
		Transmitting <= '0';
				
    elsif rising_edge(Clk) then
	  
      case present_state is
	  		 
		  -- Idle sets the out bit to '1', and waits for data, then makes a read request	 
        when Idle =>
          Reset_Counter <= '1';
			 DataOut <='1';
			 ReadRequest<='0';
			 Transmitting <= '0';
			 if DataReady = '1' then
				present_state <= S_StartBit;
				ReadRequest <= '1';
            Reset_Counter <= '0';
            DataOut <= '0';
			 end if;
  		   
		  -- generates a start bit, and lowers the read request	
        when S_StartBit =>
		    Transmitting <= '1';
	       DataOut <= '0';
          ReadRequest <='0';
     	         
			 if CounterOut > to_unsigned(wait_period,16)-2  then
			   Reset_Counter <= '1';
				present_state <= Sending;
				DataBit <= "0000";
                                DataOut <= DataIn(0); 
			 end if;
	  	    	
        -- push the 9 bites from the parallel input to the output
        when Sending =>
		    Reset_Counter <= '0';
		    Transmitting <= '1';
		    if (CounterOut > to_unsigned(wait_period,16)-3 ) then
	         Reset_Counter <= '1';
				if DataBit = "1000" then -- last bit is send
				  present_state <= EndState; 
				else
                   DataOut <= DataIn(to_integer(DataBit+1));
	           DataBit <= DataBit + 1;
				end if;
			 end if;
		    
 		  -- set a '1' stop bit, 		
        when EndState =>
		    Reset_Counter <='0';
			 Transmitting <= '1';
          DataOut <= '1';
			 if (CounterOut > to_unsigned(wait_period,16)-3 ) then
	         Reset_Counter <= '1';
				present_state <= idle;
			 end if;			 
        	
      end case; -- end state machine checks
	
	  
			 
    end if; -- end rising edge trigger
		  
		  
  end process;

end;
