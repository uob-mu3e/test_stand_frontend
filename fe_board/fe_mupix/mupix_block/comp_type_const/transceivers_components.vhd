---------------------------------------
--
-- On detector FPGA for layer 0 - transceiver component library
-- Sebastian Dittmeier, July 2016
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------



library ieee;
use ieee.std_logic_1164.all;

package transceiver_components is

component fastlink
	PORT
	(
		 cal_blk_clk	:	IN  STD_LOGIC := '0';
		 pll_inclk	:	IN  STD_LOGIC := '0';
		 pll_locked	:	OUT  STD_LOGIC_VECTOR (0 DOWNTO 0);
		 pll_powerdown	:	IN  STD_LOGIC_VECTOR (0 DOWNTO 0) := (OTHERS => '0');
		 reconfig_clk	:	IN  STD_LOGIC := '0';
		 reconfig_fromgxb	:	OUT  STD_LOGIC_VECTOR (16 DOWNTO 0);
		 reconfig_togxb	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => 'Z');
		 rx_analogreset	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 rx_clkout	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 rx_ctrldetect	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_datain	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => 'Z');
		 rx_dataout	:	OUT  STD_LOGIC_VECTOR (127 DOWNTO 0);
		 rx_digitalreset	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 rx_disperr	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_enapatternalign	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 rx_errdetect	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_freqlocked	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 rx_patterndetect	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_pll_locked	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 rx_seriallpbken	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');		 
		 rx_syncstatus	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 tx_clkout	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 tx_ctrlenable	:	IN  STD_LOGIC_VECTOR (15 DOWNTO 0) := (OTHERS => '0');
		 tx_datain	:	IN  STD_LOGIC_VECTOR (127 DOWNTO 0) := (OTHERS => '0');
		 tx_dataout	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 tx_digitalreset	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0')
	);
end component;


component transceiver_pod
	PORT
	(
		 cal_blk_clk	:	IN  STD_LOGIC := '0';
		 pll_inclk	:	IN  STD_LOGIC := '0';
		 pll_locked	:	OUT  STD_LOGIC_VECTOR (1 DOWNTO 0);
		 pll_powerdown	:	IN  STD_LOGIC_VECTOR (1 DOWNTO 0) := (OTHERS => '0');
		 reconfig_clk	:	IN  STD_LOGIC := '0';
		 reconfig_fromgxb	:	OUT  STD_LOGIC_VECTOR (33 DOWNTO 0);
		 reconfig_togxb	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => 'Z');
		 rx_analogreset	:	IN  STD_LOGIC_VECTOR (7 DOWNTO 0) := (OTHERS => '0');
		 rx_clkout	:	OUT  STD_LOGIC_VECTOR (7 DOWNTO 0);
		 rx_datain	:	IN  STD_LOGIC_VECTOR (7 DOWNTO 0) := (OTHERS => 'Z');
		 rx_dataout	:	OUT  STD_LOGIC_VECTOR (255 DOWNTO 0);
		 rx_digitalreset	:	IN  STD_LOGIC_VECTOR (7 DOWNTO 0) := (OTHERS => '0');
		 rx_enapatternalign	:	IN  STD_LOGIC_VECTOR (7 DOWNTO 0) := (OTHERS => '0');
		 rx_freqlocked	:	OUT  STD_LOGIC_VECTOR (7 DOWNTO 0);
		 rx_patterndetect	:	OUT  STD_LOGIC_VECTOR (31 DOWNTO 0);
		 rx_pll_locked	:	OUT  STD_LOGIC_VECTOR (7 DOWNTO 0);
		 rx_seriallpbken	:	IN  STD_LOGIC_VECTOR (7 DOWNTO 0) := (OTHERS => '0');		 
		 rx_syncstatus	:	OUT  STD_LOGIC_VECTOR (31 DOWNTO 0);
		 tx_clkout	:	OUT  STD_LOGIC_VECTOR (7 DOWNTO 0);
		 tx_ctrlenable	:	IN  STD_LOGIC_VECTOR (31 DOWNTO 0) := (OTHERS => '0');
		 tx_datain	:	IN  STD_LOGIC_VECTOR (255 DOWNTO 0) := (OTHERS => '0');
		 tx_dataout	:	OUT  STD_LOGIC_VECTOR (7 DOWNTO 0);
		 tx_digitalreset	:	IN  STD_LOGIC_VECTOR (7 DOWNTO 0) := (OTHERS => '0');
		 
		 rx_errdetect	:	OUT  STD_LOGIC_VECTOR (31 DOWNTO 0);
		 rx_disperr	:	OUT  STD_LOGIC_VECTOR (31 DOWNTO 0);
		 rx_ctrldetect	:	OUT  STD_LOGIC_VECTOR (31 DOWNTO 0)		 
	);
end component;


component transceiver_pod_small
	PORT
	(
		 cal_blk_clk	:	IN  STD_LOGIC := '0';
		 pll_inclk	:	IN  STD_LOGIC := '0';
		 pll_locked	:	OUT  STD_LOGIC_VECTOR (0 DOWNTO 0);
		 pll_powerdown	:	IN  STD_LOGIC_VECTOR (0 DOWNTO 0) := (OTHERS => '0');
		 reconfig_clk	:	IN  STD_LOGIC := '0';
		 reconfig_fromgxb	:	OUT  STD_LOGIC_VECTOR (16 DOWNTO 0);
		 reconfig_togxb	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => 'Z');
		 rx_analogreset	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 rx_clkout	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 rx_datain	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => 'Z');
		 rx_dataout	:	OUT  STD_LOGIC_VECTOR (127 DOWNTO 0);
		 rx_digitalreset	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 rx_enapatternalign	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 rx_freqlocked	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 rx_patterndetect	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_pll_locked	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 rx_seriallpbken	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');		 
		 rx_syncstatus	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 tx_clkout	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 tx_ctrlenable	:	IN  STD_LOGIC_VECTOR (15 DOWNTO 0) := (OTHERS => '0');
		 tx_datain	:	IN  STD_LOGIC_VECTOR (127 DOWNTO 0) := (OTHERS => '0');
		 tx_dataout	:	OUT  STD_LOGIC_VECTOR (3 DOWNTO 0);
		 tx_digitalreset	:	IN  STD_LOGIC_VECTOR (3 DOWNTO 0) := (OTHERS => '0');
		 
		 rx_errdetect	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_disperr	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0);
		 rx_ctrldetect	:	OUT  STD_LOGIC_VECTOR (15 DOWNTO 0)		 		 
	);
end component;


component gx_reset_tx
generic (
	NCHANNELS : integer := 1;
	NPLLS		 : integer := 1
	);
port(
		reset_n				: IN STD_LOGIC;	-- active low
		clk_50				: IN STD_LOGIC;	-- 50 MHz
		pll_locked			: IN STD_LOGIC_VECTOR(NPLLS-1 downto 0);	
		pll_powerdown		: OUT STD_LOGIC_VECTOR(NPLLS-1 downto 0);
		tx_digitalreset	: OUT STD_LOGIC_VECTOR(NCHANNELS-1 downto 0)
 );
end component;


component gx_reset_rx
port(
		reset_n				: IN STD_LOGIC;	-- active low
		clk_50				: IN STD_LOGIC;	-- 50 MHz
		busy					: IN STD_LOGIC;
		rx_freqlocked		: IN STD_LOGIC;
		force_digitalreset: IN STD_LOGIC;		
		rx_analogreset		: OUT STD_LOGIC;
		rx_digitalreset	: OUT STD_LOGIC
 );
end component;


--component gx_reconfig
--port (
--		reconfig_clk	: IN STD_LOGIC ;
--		reconfig_fromgxb	: IN STD_LOGIC_VECTOR (67 DOWNTO 0);
--		busy	: OUT STD_LOGIC ;
--		reconfig_togxb	: OUT STD_LOGIC_VECTOR (3 DOWNTO 0)
--);
--end component;

component gx_reconfig
port (
		reconfig_reset	: IN STD_LOGIC ;
		logical_channel_address		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		read		: IN STD_LOGIC ;
		reconfig_clk		: IN STD_LOGIC ;
		reconfig_fromgxb		: IN STD_LOGIC_VECTOR (67 DOWNTO 0);
		rx_eqctrl		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_eqdcgain		: IN STD_LOGIC_VECTOR (2 DOWNTO 0);
		tx_preemp_0t		: IN STD_LOGIC_VECTOR (4 DOWNTO 0);
		tx_preemp_1t		: IN STD_LOGIC_VECTOR (4 DOWNTO 0);
		tx_preemp_2t		: IN STD_LOGIC_VECTOR (4 DOWNTO 0);
		tx_vodctrl		: IN STD_LOGIC_VECTOR (2 DOWNTO 0);
		write_all		: IN STD_LOGIC ;
		busy		: OUT STD_LOGIC ;
		data_valid		: OUT STD_LOGIC ;
		reconfig_togxb		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_eqctrl_out		: OUT STD_LOGIC_VECTOR (3 DOWNTO 0);
		rx_eqdcgain_out		: OUT STD_LOGIC_VECTOR (2 DOWNTO 0);
		tx_preemp_0t_out		: OUT STD_LOGIC_VECTOR (4 DOWNTO 0);
		tx_preemp_1t_out		: OUT STD_LOGIC_VECTOR (4 DOWNTO 0);
		tx_preemp_2t_out		: OUT STD_LOGIC_VECTOR (4 DOWNTO 0);
		tx_vodctrl_out		: OUT STD_LOGIC_VECTOR (2 DOWNTO 0);
		error	: OUT STD_LOGIC ;

		reconfig_mode_sel		: IN STD_LOGIC_VECTOR (3 DOWNTO 0);		
		ctrl_address		: IN STD_LOGIC_VECTOR (15 DOWNTO 0);
		ctrl_read		: IN STD_LOGIC ;
		ctrl_write		: IN STD_LOGIC ;
		ctrl_writedata		: IN STD_LOGIC_VECTOR (15 DOWNTO 0);
		ctrl_readdata		: OUT STD_LOGIC_VECTOR (15 DOWNTO 0);
		ctrl_waitrequest		: OUT STD_LOGIC

);
end component;

end package transceiver_components;