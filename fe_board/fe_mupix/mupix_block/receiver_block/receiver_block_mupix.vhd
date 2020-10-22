-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Receiver block for all the LVDS links
-- Niklaus Berger, May 2013
-- 
-- nberger@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.lvds_components.all;
use work.datapath_components.all;
use work.detectorfpga_types.all;


entity receiver_block is 
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
		rx_clkout			: out std_logic_vector(2 downto 0);
		rx_doubleclk		: out std_logic_vector(1 downto 0);
		pll_locked			: out std_logic_vector(2 downto 0)
		
		);
end receiver_block;		
		
architecture rtl of receiver_block is

	signal rx_out : 			std_logic_vector(NINPUT*10-1 downto 0);
	signal rx_out_temp : 	std_logic_vector(NINPUT*10-1 downto 0);	
	signal rx_clk :			std_logic_vector(2 downto 0);

	signal rx_dpa_locked		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_align			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_fifo_reset		: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	signal rx_reset			: STD_LOGIC_VECTOR (NINPUT-1 DOWNTO 0);
	
	signal rx_locked			: STD_LOGIC_VECTOR(2 downto 0);
	
	signal 	rx_inclock_A_pll:		std_logic;
	signal 	rx_enable_A:			std_logic;
	signal   rx_synclock_A:			std_logic;	
	signal 	rx_dpaclock_A:			std_logic;
	
	signal 	rx_inclock_B_pll:		std_logic;
	signal 	rx_enable_B:			std_logic;
	signal   rx_synclock_B:			std_logic;	
	signal 	rx_dpaclock_B:			std_logic;	
	
	signal 	rx_inclock_FF_pll:	std_logic;
	signal 	rx_enable_FF:			std_logic;
	signal   rx_synclock_FF:		std_logic;	
	signal 	rx_dpaclock_FF:		std_logic;		
	signal 	rx_dpa_locked_FF	: STD_LOGIC_VECTOR (1 DOWNTO 0);
	signal 	rx_align_FF			: STD_LOGIC_VECTOR (1 DOWNTO 0);
	signal 	rx_fifo_reset_FF	: STD_LOGIC_VECTOR (1 DOWNTO 0);
	signal 	rx_reset_FF			: STD_LOGIC_VECTOR (1 DOWNTO 0);
	
	signal rx_inclock_A_ctrl	: std_logic;
	
	signal	rx_ready_FF:		std_logic_vector(1 downto 0);	
	signal	rx_k_FF:				std_logic_vector(1 downto 0);	
	signal	rx_state_FF:		std_logic_vector(7 downto 0);
	signal	rx_out_FF			: std_logic_vector(19 downto 0);	
	
	
	signal decoder_ena:	std_logic_vector(NINPUT-1 downto 0);

begin

	pll_locked <= rx_locked;
	rx_clk(0) <= rx_synclock_A;
	rx_clk(1) <= rx_synclock_B;	
	rx_clk(2) <= rx_synclock_FF;
	rx_clkout <= rx_clk;

	--rx_fifo_reset		<= rx_reset;

ctrl_test: sync_clkctrl 
	port map (
		inclk  => rx_inclock_A,
		outclk => rx_inclock_A_ctrl
	);	
	
lpll_A:lvdspll
	PORT MAP
	(
		refclk   => rx_inclock_A_ctrl,
		rst      => '0',
		outclk_0 => rx_inclock_A_pll,
		outclk_1 => rx_enable_A,
		outclk_2 => rx_synclock_A,
		outclk_3 => rx_dpaclock_A,		
		outclk_4 => rx_doubleclk(0),
		locked   => rx_locked(0)	
);	
--
----syncclkctrl_A:sync_clkctrl 
----	port map (
----			inclk  => rx_synclock_A, -- inclk
----			outclk => rx_clk(0)	   -- outclk
----		);
--		
lvds_rec_small:lvds_receiver
	PORT MAP
	(
		pll_areset						=> not rx_locked(0),
		rx_channel_data_align		=> rx_align(26 downto 0),
		rx_dpaclock						=> rx_dpaclock_A,
		rx_enable						=> rx_enable_A,
		rx_fifo_reset					=> rx_fifo_reset(26 downto 0),
		rx_in								=> rx_in(26 downto 0),
		rx_inclock						=> rx_inclock_A_pll,
		rx_reset							=> rx_reset(26 downto 0),
		rx_syncclock					=> rx_synclock_A,
		rx_dpa_locked					=> rx_dpa_locked(26 downto 0),
		rx_out							=> rx_out(269 downto 0)
	);
	
	

lpll_B:lvdspll
	PORT MAP
	(
		refclk   => rx_inclock_B,
		rst      => '0',
		outclk_0 => rx_inclock_B_pll,
		outclk_1 => rx_enable_B,
		outclk_2 => rx_synclock_B,
		outclk_3 => rx_dpaclock_B,		
		outclk_4 => rx_doubleclk(1),
		locked   => rx_locked(1)
);

--syncclkctrl_B:sync_clkctrl 
--	port map (
--			inclk  => rx_synclock_B, -- inclk
--			outclk => rx_clk(1)  -- outclk
--		);
	
	
--lvds_rec:lvds_receiver
--	PORT MAP
--	(
--		pll_areset					=> not rx_locked(1),
--		rx_channel_data_align 	=> rx_align(44 downto 18),
--		rx_enable					=> rx_enable_B,
--		rx_fifo_reset				=> rx_fifo_reset(44 downto 18),
--		rx_in							=> rx_in(44 downto 18),
--		rx_inclock		   		=> rx_inclock_B_pll,
--		rx_reset						=> rx_reset(44 downto 18),
--		rx_syncclock				=> rx_synclock_B,
--		rx_dpa_locked				=> rx_dpa_locked(44 downto 18),
--		rx_out						=> rx_out(449 downto 180)
--	);
		
	
lvds_rec:lvds_receiver_small
	PORT MAP
	(
		pll_areset					=> not rx_locked(1),
		rx_channel_data_align 	=> rx_align(35 downto 27),
		rx_dpaclock					=> rx_dpaclock_B,		
		rx_enable					=> rx_enable_B,
		rx_fifo_reset				=> rx_fifo_reset(35 downto 27),
		rx_in							=> rx_in(35 downto 27),
		rx_inclock		   		=> rx_inclock_B_pll,
		rx_reset						=> rx_reset(35 downto 27),
		rx_syncclock				=> rx_synclock_B,
		rx_dpa_locked				=> rx_dpa_locked(35 downto 27),
		rx_out						=> rx_out_temp(359 downto 270)
	);
	
	-- Input D9 is inverted...
	rx_out(359 downto 350) <= not rx_out_temp(359 downto 350);
	rx_out(349 downto 270)   <= rx_out_temp(349 downto 270);
	
			
--lpll_FF:lvdspll
--	PORT MAP
--	(
--		refclk   => rx_inclock_A_ctrl,
--		rst      => '0',
--		outclk_0 => rx_inclock_FF_pll,
--		outclk_1 => rx_enable_FF,
--		outclk_2 => rx_synclock_FF,
--		outclk_3 => rx_dpaclock_FF,		
--		outclk_4 => open,
--		locked   => rx_locked(2)	
--);		
	

		
		
resgen:
FOR i in NCHIPS-1 downto 0 generate		
	chip_reset(i) <= '1' when rx_reset(3*i) = '1' or 
									  rx_reset(3*i+1) = '1' or 
									  rx_reset(3*i+2) = '1'
						else '0';
END GENERATE;	

	
gendec:
FOR i in NINPUT-1 downto 0 generate	

datadec: data_decoder 
	port map(
		reset_n				=> reset_n,
		checker_rst_n		=> checker_rst_n(i),
		clk					=> rx_clk(i/27),
		rx_in					=> rx_out(i*10+9 downto i*10),
		
		rx_reset				=> rx_reset(i),
		rx_fifo_reset     => rx_fifo_reset(i),
		rx_dpa_locked		=> rx_dpa_locked(i),
		rx_locked			=> rx_locked(i/26),
		rx_align				=> rx_align(i),
	
		ready					=> rx_ready(i),
		data					=> rx_data(i),
		k						=> rx_k(i),
		state_out			=> rx_state(i*4+3 downto i*4)
		);


end generate;


end rtl; 
