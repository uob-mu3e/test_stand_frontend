---------------------------------------
--
-- On detector FPGA for layer 0 - lvds component library
-- Sebastian Dittmeier, July 2016
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;
use work.detectorfpga_types.all;

package lvds_components is


component receiver_block_mupix
	generic(
		NINPUT: integer := 45;
		NCHIPS: integer := 15
	);
	port (
		reset_n				: in std_logic;
		checker_rst_n		: in std_logic_vector(NINPUT-1 downto 0);
		rx_in					: IN STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
		rx_inclock_A		: IN STD_LOGIC ;
		rx_inclock_B		: IN STD_LOGIC ;
		chip_reset			: out std_logic_vector(NCHIPS-1 downto 0);
		rx_state				: out std_logic_vector(NINPUT*4-1 downto 0);
		rx_ready				: out std_logic_vector(NINPUT-1 downto 0);
		rx_data				: out inbyte_array;
		rx_k					: out std_logic_vector(NINPUT-1 downto 0);
		rx_clkout			: out std_logic_vector(1 downto 0);
		rx_doubleclk		: out std_logic_vector(1 downto 0);
		pll_locked			: out std_logic_vector(1 downto 0)
		);
end component;		

component lvds_receiver_mupix
	PORT
	(
		pll_areset		: IN STD_LOGIC ;
		rx_channel_data_align		: IN STD_LOGIC_VECTOR (26 DOWNTO 0);
		rx_enable		: IN STD_LOGIC ;
		rx_fifo_reset		: IN STD_LOGIC_VECTOR (26 DOWNTO 0);
		rx_in		: IN STD_LOGIC_VECTOR (26 DOWNTO 0);
		rx_inclock		: IN STD_LOGIC ;
		rx_reset		: IN STD_LOGIC_VECTOR (26 DOWNTO 0);
		rx_syncclock		: IN STD_LOGIC ;
		rx_dpa_locked		: OUT STD_LOGIC_VECTOR (26 DOWNTO 0);
		rx_out		: OUT STD_LOGIC_VECTOR (269 DOWNTO 0)
	);
end component;


component lvds_receiver_small
	PORT
	(
		pll_areset		: IN STD_LOGIC ;
		rx_channel_data_align		: IN STD_LOGIC_VECTOR (17 DOWNTO 0);
		rx_enable		: IN STD_LOGIC ;
		rx_fifo_reset		: IN STD_LOGIC_VECTOR (17 DOWNTO 0);
		rx_in		: IN STD_LOGIC_VECTOR (17 DOWNTO 0);
		rx_inclock		: IN STD_LOGIC ;
		rx_reset		: IN STD_LOGIC_VECTOR (17 DOWNTO 0);
		rx_syncclock		: IN STD_LOGIC ;
		rx_dpa_locked		: OUT STD_LOGIC_VECTOR (17 DOWNTO 0);
		rx_out		: OUT STD_LOGIC_VECTOR (179 DOWNTO 0)
	);
end component;

component lvdspll
	PORT
	(
		inclk0		: IN STD_LOGIC  := '0';
		c0		: OUT STD_LOGIC ;
		c1		: OUT STD_LOGIC ;
		c2		: OUT STD_LOGIC ;
		c3		: OUT STD_LOGIC ;
		locked		: OUT STD_LOGIC 
	);
end component;

Component data_decoder_mupix
	port (
		reset_n				: in std_logic;
		checker_rst_n		: in std_logic;
		clk					: in std_logic;
		rx_in					: IN STD_LOGIC_VECTOR (9 DOWNTO 0);
		
		rx_reset				: OUT STD_LOGIC;
		rx_fifo_reset		: OUT STD_LOGIC;
		rx_dpa_locked		: IN STD_LOGIC;
		rx_locked			: IN STD_LOGIC;
		rx_align				: OUT STD_LOGIC;
	
		ready					: OUT STD_LOGIC;
		data					: OUT STD_LOGIC_VECTOR(7 downto 0);
		k						: OUT STD_LOGIC;
		state_out			: OUT STD_LOGIC_VECTOR(3 downto 0)
		);
end component;


component dec_8b10b 	
    port(
		RESET : in std_logic ;	-- Global asynchronous reset (AH) 
		RBYTECLK : in std_logic ;	-- Master synchronous receive byte clock
		AI, BI, CI, DI, EI, II : in std_logic ;
		FI, GI, HI, JI : in std_logic ; -- Encoded input (LS..MS)		
		KO : out std_logic ;	-- Control (K) character indicator (AH)
		HO, GO, FO, EO, DO, CO, BO, AO : out std_logic 	-- Decoded out (MS..LS)
	    );
end component;

component decode8b10b 
	port (
		reset_n				: in std_logic;
		clk					: in std_logic;
		input					: IN STD_LOGIC_VECTOR (9 DOWNTO 0);
		output				: OUT STD_LOGIC_VECTOR (7 DOWNTO 0);
		k						: OUT std_logic
		);
end component;

end package lvds_components;