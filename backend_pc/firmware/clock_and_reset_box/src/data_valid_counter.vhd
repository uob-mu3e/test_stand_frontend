library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
Library UNISIM;
use UNISIM.vcomponents.all;
entity data_valid_counter is
	port(
		clk : in std_logic;
		data: in std_logic_vector(31 downto 0);
		data_valid: out std_logic;
		rst : in std_logic
	);
end entity data_valid_counter;

architecture RTL of data_valid_counter is
	signal Q1,Q2,Q3,Q4,Q5 : std_ulogic;
	signal data_is_k: std_ulogic;
	signal sync_out: std_ulogic;
begin
   SYNC1 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q1,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => rst,  -- Asynchronous clear input
      D => data_is_k       -- Data input
   );
   SYNC2 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q2,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => rst,  -- Asynchronous clear input
      D => Q1       -- Data input
   );
   
   SYNC3 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q3,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => rst,  -- Asynchronous clear input
      D => Q2       -- Data input
   );
   
   SYNC4 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q4,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => rst,  -- Asynchronous clear input
      D => Q3       -- Data input
   );
   
   SYNC5 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q5,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => rst,  -- Asynchronous clear input
      D => Q4       -- Data input
   );
   
   sync_mon : process (clk, rst) is
   begin
   	if rst = '1' then
   		sync_out <= '0';
   	elsif rising_edge(clk) then
   		if (Q1 = '1' and Q2='1' and Q3='1' and Q4 = '1' and Q5 = '1') then
   			sync_out <= '1';
   		elsif (Q1 = '0' and Q2='0' and Q3='0' and Q4 = '0' and Q5 = '0') then
   			sync_out <= '0';
   		end if;
   	end if;
   end process sync_mon;
   
   data_valid <= sync_out;
   data_is_k <= '1' when data = X"bcbcbcbc" else '0';
end architecture RTL;
