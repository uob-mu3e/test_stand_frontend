--********************************************************************************************************************--
--! @file
--! @brief File Description
--! Copyright&copy - YOUR COMPANY NAME
--********************************************************************************************************************--
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

--! Local libraries
library work;

--! Entity/Package Description
entity TOP_MU3E is
   port (	TX_MGT_RST_L: 	out std_logic;
			TX_MGT_INT_L: 	in	std_logic;
			TX_MGT_SLCT_L: 	out std_logic;
			TX_MGT_PRSNT_L: in	std_logic;
			FF_RX_INT_L: 	in	std_logic;
			FF_TX_INT_L: 	in	std_logic;
			TX_CLK_RST_L: 	out std_logic;
			TX_CLK_INT_L: 	in	std_logic;
			TX_CLK_SLCT_L: 	out std_logic;
			TX_CLK_PRSNT_L: in	std_logic;
			CLK_INTR_L: 	in	std_logic;
			CLK_OE_L: 		out std_logic;
			CLK_RST_L: 		out std_logic;
			CLK_LOL_L: 		in	std_logic

   );
end entity TOP_MU3E;

architecture rtl of TOP_MU3E is

begin
			TX_MGT_RST_L	<= '1';
			TX_MGT_SLCT_L 	<= '1';
			TX_CLK_RST_L	<= '1';
			TX_CLK_SLCT_L 	<= '1';
			CLK_OE_L  		<= '0';
			CLK_RST_L 		<= '1';
end architecture rtl;


