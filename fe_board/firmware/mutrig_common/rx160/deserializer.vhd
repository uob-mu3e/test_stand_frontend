library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use work.rx160_components.all;

entity deserializer is
  port (
    o_sync_found  : out std_logic;		               -- debug output to indicate sync was found
    --i_start_val  : in  std_logic_vector(3 downto 0); -
    i_ser_clk     : in  std_logic;			            -- ser 160Mhz clock
    i_rst         : in  std_logic;			            -- reset active high
    i_serial_data : in  std_logic;                    -- the serial sync data
    o_byte_clock  : out std_logic;		               -- the slow byte clock
    o_byte_data   : out std_logic_vector(7 downto 0); -- the byte data
    o_kontrol     : out std_logic                     -- k control word flag
  );
end entity deserializer;


architecture RTL of deserializer is

	signal fast_clk : std_logic := '0';
	signal byte_data : std_logic_vector(7 downto 0);
	signal slow_clock : std_logic;
	signal local_data : std_logic_vector(9 downto 0);
	signal local_sampled_data : std_logic_vector(9 downto 0);
	signal rst : std_logic;
	signal s_o_kontrol : std_logic;
	signal sync_found : std_logic; 

	signal no_sync_since_reset : std_logic; -- HEADER ID 10 BIT 1100001011
	CONSTANT komma_sync : std_logic_vector(9 downto 0) := "0101111100"; --K28.5

	signal err_found : std_logic;
	signal reset_dec : std_logic;

begin

	--RESET THE DECODER IF AN ERRONEOUS CODE WAS FOUND OR IF A BUS RESET OCCURED
	reset_dec <= err_found or i_rst;

	o_sync_found <= sync_found;

	o_kontrol <= s_o_kontrol;
	o_byte_data <= byte_data;
	o_byte_clock <= slow_clock;	--AT SOME POINT I SHOULD GIVE ALL SIGNALS UNIFORM NAMES... 

decoder: dec_8b10b
  port map(
    RESET => no_sync_since_reset,
    RBYTECLK => slow_clock,
    AI=> local_sampled_data(0),
    BI=> local_sampled_data(1),
    CI=> local_sampled_data(2),
    DI=> local_sampled_data(3),
    EI=> local_sampled_data(4),
    II=> local_sampled_data(5),
    FI=> local_sampled_data(6),
    GI=> local_sampled_data(7),
    HI=> local_sampled_data(8),
    JI=> local_sampled_data(9),
    KO => s_o_kontrol,
    AO => byte_data(0),
    BO => byte_data(1),
    CO => byte_data(2),
    DO => byte_data(3),
    EO => byte_data(4),
    FO => byte_data(5),
    GO => byte_data(6),
    HO => byte_data(7)
);

byte_clk: bclock_gen
  port map(
    i_clk => i_ser_clk,
    --i_start_val => i_start_val,
    i_rst => sync_found,
    o_div_clk => slow_clock
);

-- shiftregister for the incoming data, sync to falling edge
sample_data : process (i_ser_clk) 
  begin
    if falling_edge(i_ser_clk) then
      local_data <= i_serial_data & local_data(local_data'high downto 1);
    end if;
end process ;

-- error detection, singal lost
check_err : process(i_ser_clk)
  begin
    if rising_edge(i_ser_clk) then
      -- 6 subsequent 1 or 0 should never occure
      if (local_data(9 downto 3) = "0000000" or local_data(9 downto 3) = "1111111") then
        err_found <= '1';
      else
        err_found <= '0';
      end if;
    end if;
  end process;

 -- find sync signals, used to start start/sync slow byte clock
find_syn : process (i_ser_clk)
  begin
    if rising_edge(i_ser_clk) then
      if (local_data(9 downto 0) = "1010000011" or local_data(9 downto 0) = "0101111100") then
        sync_found <= '1';
      else
        sync_found <= '0';
      end if;
    end if;
   end process find_syn;

-- sample the 10b data with sync slow byte clock
sample_input : process (slow_clock)
  begin
    if rising_edge(slow_clock) then	
      local_sampled_data <= local_data;
    end if;
  end process sample_input;

 -- reset for 8b140b decoder 
reset_encoder : process(i_ser_clk)
begin
  if rising_edge(i_ser_clk) then
    if reset_dec = '1' then
      no_sync_since_reset <= '1';
    elsif sync_found = '1' then
      no_sync_since_reset <= '0';
    end if;
  end if;
end process;

end architecture RTL ;	
