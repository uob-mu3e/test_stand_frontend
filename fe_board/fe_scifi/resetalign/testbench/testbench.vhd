-- interface for talking to stratix4 io delay chains (level one delay)

Library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.STD_LOGIC_ARITH.all;
use IEEE.STD_LOGIC_UNSIGNED.all;


entity testbench is
end testbench; 


architecture tb of testbench is
component clockalign_block is
	generic(
			CLKDIV  : integer := 125--;
	);
	port(
		--SYSTEM SIGNALS
		i_clk_config : in std_logic;
		i_rst : in std_logic;

		i_flag  : in std_logic;
		i_data  : in std_logic_vector(31 downto 0);

		i_pll_clk : in std_logic;
		i_pll_arst : in std_logic;
		o_pll_clk : out std_logic_vector(3 downto 0);
		o_pll_locked : out std_logic;

		i_sig : in std_logic;
		o_sig : out std_logic_vector(3 downto 0)
	);
end component; -- resetbuf_block; 

signal i_clk : std_logic:='0';
signal i_rst : std_logic:='0';
signal i_arst : std_logic:='0';

signal i_flag : std_logic:='0';
signal i_data: std_logic_vector(31 downto 0); 

signal sigin : std_logic:='1';
signal sigout : std_logic_vector(3 downto 0);
signal sigoutx : std_logic;
signal clkout : std_logic_vector(3 downto 0);

begin
	u_dut: clockalign_block
	generic map (CLKDIV => 2)
	port map(
		i_clk_config => i_clk,
		i_rst        => i_rst,
                                          
		i_pll_clk    => i_clk,
		i_pll_arst   => i_rst,

		i_flag       => i_flag,
		i_data       => i_data,
                                          
		i_sig => sigin,
		o_sig => sigout,
		o_pll_clk    => clkout
	 ); 



	i_clk <= not i_clk after 2.5 ns;
	sigin <= not sigin after 25 ns;
	sigoutx <= sigout(0) xor sigout(1);
	p_rst: process
	begin
		wait for 10 ns;
		i_rst<= '1';
		wait for 10 ns;
		wait until rising_edge(i_clk);
		i_rst<='0';
		wait;
	end process;

	p_stim: process
	begin
		wait for 100 ns;
		wait until rising_edge(i_clk);
		i_data <= x"ffffff"&"00000101";
		i_flag <= '1';
		wait until rising_edge(i_clk);
		i_flag <= '0';
	end process;
end architecture;

