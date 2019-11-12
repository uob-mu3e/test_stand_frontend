--RST must be asserted for five read and write clock

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
Library UNISIM;
use UNISIM.vcomponents.all;
entity fifo_reset_ctrl is
	port(
		clk : in std_logic; --the slowest clock be it RDCLK or WRCLK
		rsti : in std_logic;
		rsto :  out std_logic
	);
end entity fifo_reset_ctrl;

architecture RTL of fifo_reset_ctrl is
--	signal rsti_b : std_logic;
--	signal go : std_logic;
	signal Q1,Q2,Q3,Q4,Q5 : std_ulogic;


begin

   SYNCRST1 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q1,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => '0',  -- Asynchronous clear input
      D => rsti       -- Data input
   );
   SYNCRST2 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q2,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => '0',  -- Asynchronous clear input
      D => Q1       -- Data input
   );
   
   SYNCRST3 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q3,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => '0',  -- Asynchronous clear input
      D => Q2       -- Data input
   );
   
   SYNCRST4 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q4,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => '0',  -- Asynchronous clear input
      D => Q3       -- Data input
   );
   
   SYNCRST5 : FDCE
   generic map (
      INIT => '0') -- Initial value of register ('0' or '1')  
   port map (
      Q => Q5,      -- Data output
      C => clk,      -- Clock input
      CE => '1',    -- Clock enable input
      CLR => '0',  -- Asynchronous clear input
      D => Q4       -- Data input
   );


--rst_edge_detect : process (clk) is
--begin
--	if rising_edge(clk) then
--		if (rsti_b /= rsti) then
--			go <= '1';
--		else
--			go <= '0';
--		end if;
--		rsti_b <= rsti;
--	end if;
--end process rst_edge_detect;

rsto <= Q1 or Q2 or Q3 or Q4 or Q5;

end architecture RTL;
