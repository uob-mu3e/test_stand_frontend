library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.pcie_components.all;
use work.mudaq_registers.all;
use work.mudaq_components.all;


entity top is
port (
    BUTTON : in std_logic_vector(3 downto 0);

    CLK_50_B2J  :   in  std_logic;

    HEX0_D      :   out std_logic_vector(6 downto 0);
    --HEX0_DP     :   out std_logic;

    HEX1_D      :   out std_logic_vector(6 downto 0);
    --HEX1_DP     :   out std_logic;

    LED         :   out std_logic_vector(3 downto 0);
    LED_BRACKET :   out std_logic_vector(3 downto 0);

    SMA_CLKOUT : out std_logic;
    SMA_CLKIN : in std_logic;

    RS422_DE : out std_logic;
    RS422_DIN : in std_logic; -- 1.8-V
    RS422_DOUT : out std_logic;
    RS422_RE_n : out std_logic;
    RJ45_LED_L : out std_logic;
    RJ45_LED_R : out std_logic;
	 
	 refclk2_qr1_p	: in std_logic;--					1.5-V PCML, default 125MHz
	 refclk1_qr0_p : in std_logic;-- 1.5-V PCML, default 156.25MHz
	 
	  --      ///////// FAN /////////
    FAN_I2C_SCL :   out     std_logic;
    FAN_I2C_SDA :   inout   std_logic;
	 
	 
	 --     ///////// CPU /////////
    CPU_RESET_n :   in  std_logic;
	 
	 --      ///////// FLASH /////////
	 FLASH_A         :   out     std_logic_vector(26 downto 1);
    FLASH_D         :   inout   std_logic_vector(31 downto 0);
    FLASH_OE_n      :   inout   std_logic;
    FLASH_WE_n      :   out     std_logic;
    FLASH_CE_n      :   out     std_logic_vector(1 downto 0);
    FLASH_ADV_n     :   out     std_logic;
    FLASH_CLK       :   out     std_logic;
    FLASH_RESET_n   :   out     std_logic;
	 
	 --      ///////// POWER /////////
    POWER_MONITOR_I2C_SCL   :   out     std_logic;
    POWER_MONITOR_I2C_SDA   :   inout   std_logic;
	 
	 --      ///////// TEMP /////////
    TEMP_I2C_SCL    :   out     std_logic;
    TEMP_I2C_SDA    :   inout   std_logic;

    SW : in std_logic_vector(1 downto 0);
	 
	 --clkin_50_top	: in std_logic;--					2.5V, default 50MHz
	 
	 --///////// Transiver /////////
	 QSFPA_TX_p          :   out std_logic_vector(3 downto 0);
	 QSFPB_TX_p          :   out std_logic_vector(3 downto 0);
    
	 QSFPA_RX_p          :   in std_logic_vector(3 downto 0);
	 QSFPB_RX_p          :   in std_logic_vector(3 downto 0);
    
	 QSFPA_REFCLK_p 		: 	 in std_logic;
	 QSFPB_REFCLK_p 		: 	 in std_logic;
    
    
	
	 --///////// PCIE /////////
    PCIE_PERST_n			:	in	std_logic;
    PCIE_REFCLK_p			:	in	std_logic;
    PCIE_RX_p				:	in	std_logic_vector(7 downto 0);
    PCIE_SMBCLK			:	in	std_logic;
    PCIE_SMBDAT			:	inout	std_logic;
    PCIE_TX_p				:	out std_logic_vector(7 downto 0);
    PCIE_WAKE_n			:	out std_logic
	
	 );

end entity top;

architecture rtl of top is


		 signal clk : std_logic;
		 signal reset_n : std_logic;
		 
		 signal clk_125_cnt : std_logic_vector(31 downto 0);
		 signal clk_125_cnt2 : std_logic_vector(31 downto 0);


		------------------ Signal declaration ------------------------

		-- external connections
		--signal clk : std_logic;
		--signal xcvr_ref_clk : std_logic;	 
		--signal refclk_125 : std_logic;	 

		-- pcie registers
		signal writeregs				: reg32array;
		signal regwritten				: std_logic_vector(63 downto 0);
		signal regwritten_fast		: std_logic_vector(63 downto 0);
		signal regwritten_del1		: std_logic_vector(63 downto 0);
		signal regwritten_del2		: std_logic_vector(63 downto 0);
		signal regwritten_del3		: std_logic_vector(63 downto 0);
		signal regwritten_del4		: std_logic_vector(63 downto 0);

		signal readregs				: reg32array;
		signal readregs_slow			: reg32array;

		--//pcie readable memory signals
		signal readmem_writedata 	: std_logic_vector(31 downto 0);
		signal readmem_writeaddr 	: std_logic_vector(63 downto 0);
		signal readmem_writeaddr_lowbits : std_logic_vector(15 downto 0);
		signal readmem_wren	 		: std_logic;
		signal readmem_endofevent 	: std_logic;
		--//pcie writeable memory signals
		signal writememreadaddr 	: std_logic_vector(15 downto 0);
		signal writememreaddata 	: std_logic_vector (31 downto 0);

		--//pcie dma memory signals
		signal dmamem_writedata 	: std_logic_vector(255 downto 0);
		signal dmamem_wren	 		: std_logic;
		signal dmamem_endofevent 	: std_logic;
		signal dmamemhalffull 		: std_logic;

		--//pcie dma memory signals
		signal dma2mem_writedata 	: std_logic_vector(255 downto 0);
		signal dma2mem_wren	 		: std_logic;
		signal dma2mem_endofevent 	: std_logic;
		signal dma2memhalffull 		: std_logic;
		
		-- pcie fast clock
		signal pcie_fastclk_out		: std_logic;	

		-- pcie debug signals
		signal pcie_testout				: std_logic_vector(127 downto 0);

		-- Clocksync stuff
		signal clk50_sync : std_logic;
		signal clk50_last : std_logic;
		signal clk_sync : std_logic;
		signal clk_last : std_logic;
		signal clk156ch0_sync : std_logic;
		signal clk156ch0_last : std_logic;
		signal clk156ch1_sync : std_logic;
		signal clk156ch1_last : std_logic;
		signal clk156ch2_sync : std_logic;
		signal clk156ch2_last : std_logic;
		signal clk156ch3_sync : std_logic;
		signal clk156ch3_last : std_logic;
		
		signal reset : std_logic;
		
		signal resets: std_logic_vector(31 downto 0);
		signal resets_n: std_logic_vector(31 downto 0);
		
		signal status_rr: std_logic_vector(31 downto 0);
		
		signal reset_156 : std_logic;
		signal reset_n_156 : std_logic;
		--signal sl_test_enable : std_logic;
		--signal blinker_signal : std_logic;

		--signal push_button0_db : std_logic;
		--signal push_button1_db : std_logic;
		--signal push_button2_db : std_logic;
		
		signal input_clk : std_logic;
		signal clk_156 : std_logic;

		-- tranciever/reset ip signals
		signal rx_clkout_ch0_clk : std_logic;
		signal rx_clkout_ch1_clk : std_logic;
		signal rx_clkout_ch2_clk : std_logic;
		signal rx_clkout_ch3_clk : std_logic;
		
		signal pll_powerdown : std_logic;
		signal tx_analogreset : std_logic_vector(3 downto 0);
		signal tx_digitalreset : std_logic_vector(3 downto 0);
		signal tx_ready : std_logic_vector(3 downto 0);
		signal pll_locked : std_logic;
		signal pll_locked_vector : std_logic_vector(1 downto 0);
		signal tx_cal_busy : std_logic_vector(3 downto 0);
		signal rx_analogreset : std_logic_vector(3 downto 0);
		signal rx_digitalreset : std_logic_vector(3 downto 0);
		signal rx_ready : std_logic_vector(3 downto 0);
		signal rx_is_lockedtodata : std_logic_vector(3 downto 0);
		signal rx_is_lockedtoref : std_logic_vector(4 downto 0);
		signal rx_cal_busy : std_logic_vector(3 downto 0);

		signal tx_parallel_data : std_logic_vector(31 downto 0);
		signal rx_parallel_data : std_logic_vector(31 downto 0);

		signal tx_clk : std_logic_vector(1 downto 0);
		signal rx_clk : std_logic_vector(1 downto 0);
		signal tx_std_coreclkin : std_logic_vector(1 downto 0);
		signal rx_std_coreclkin : std_logic_vector(1 downto 0);
		signal rx_bitslip : std_logic_vector(1 downto 0);
		signal rx_serial_loopback_enable : std_logic_vector(1 downto 0);

		signal tx_clkout	  : std_logic_vector(3 downto 0);
		signal rx_clkout    : std_logic_vector(3 downto 0);

		signal tx_datak : std_logic_vector(3 downto 0);
		signal rx_datak : std_logic_vector(15 downto 0);

		signal rx_errdetect : std_logic_vector(7 downto 0);
		signal rx_disperr : std_logic_vector(7 downto 0);

		signal reconfig_to_xcvr : std_logic_vector(279 downto 0);
		signal reconfig_from_xcvr : std_logic_vector(183 downto 0);  
		signal reconfig_busy : std_logic;

		signal pll_cal_busy : std_logic;

		signal tx_serial_clk : std_logic;

		signal rx_pattern : std_logic_vector(15 downto 0);
		signal rx_syncstatus : std_logic_vector(15 downto 0);

		signal pll_conf_write       :   std_logic := '0';
		signal pll_conf_read        :   std_logic := '0';
		signal pll_conf_address     :   std_logic_vector(15 downto 0);
		signal pll_conf_writedata   :   std_logic_vector(31 downto 0);
		signal pll_conf_readdata    :   std_logic_vector(31 downto 0);
		signal pll_conf_wait        :   std_logic;
		
		signal rx_is_lockedtodata_rx_is_lockedtodata : std_logic_vector(3 downto 0);
		
		--- data processing signals
		signal s_clk : std_logic;
		signal s_cnt : std_logic_vector(31 downto 0) := (others => '0');
		signal led_changer : std_logic;
		signal data : std_logic_vector(23 downto 0);
		---signal counter : std_logic_vector(39 downto 0);

		--- config stuff
		signal clbr : std_logic;
		signal conf_state           :   integer range 0 to 31 := 0;
		signal phy_conf_write       :   std_logic := '0';
		signal phy_conf_read        :   std_logic := '0';
		signal phy_conf_channel     :   std_logic_vector(1 downto 0);
		signal phy_conf_address     :   std_logic_vector(15 downto 0);
		signal phy_conf_writedata   :   std_logic_vector(31 downto 0);
		signal phy_conf_readdata    :   std_logic_vector(31 downto 0);
		signal phy_conf_wait        :   std_logic;
		signal clbr_q0              :   std_logic;

		--- 7 segment stuff ---
		constant c_CNT_1HZ   : natural := 25000000;
		signal r_CNT_1HZ   : natural range 0 to 25000000;
		
		
		--- transiver stuff ---
		signal rx_parallel_data_ch0_rx_parallel_data :   std_logic_vector(31 downto 0);
		signal rx_parallel_data_ch1_rx_parallel_data :   std_logic_vector(31 downto 0);
		signal rx_parallel_data_ch2_rx_parallel_data :   std_logic_vector(31 downto 0);
		signal rx_parallel_data_ch3_rx_parallel_data :   std_logic_vector(31 downto 0);
		
		signal rx_datak_ch0_rx_datak      :   std_logic_vector(3 downto 0);
		signal rx_datak_ch1_rx_datak      :   std_logic_vector(3 downto 0);
		signal rx_datak_ch2_rx_datak      :   std_logic_vector(3 downto 0);
		signal rx_datak_ch3_rx_datak      :   std_logic_vector(3 downto 0);
	
		signal rx_syncstatus_ch0_rx_syncstatus:   std_logic_vector(3 downto 0);
		signal rx_syncstatus_ch1_rx_syncstatus:   std_logic_vector(3 downto 0);
		signal rx_syncstatus_ch2_rx_syncstatus:   std_logic_vector(3 downto 0);
		signal rx_syncstatus_ch3_rx_syncstatus:   std_logic_vector(3 downto 0);
		
		signal rx_patterndetect_ch0_rx_patterndetect   : std_logic_vector(3 downto 0);
		signal rx_patterndetect_ch1_rx_patterndetect   : std_logic_vector(3 downto 0);
		signal rx_patterndetect_ch2_rx_patterndetect   : std_logic_vector(3 downto 0);
		signal rx_patterndetect_ch3_rx_patterndetect   : std_logic_vector(3 downto 0);
		
		
		signal data_ch0 : std_logic_vector(31 downto 0);
		signal data_ch1 : std_logic_vector(31 downto 0);
		signal data_ch2 : std_logic_vector(31 downto 0);
		signal data_ch3 : std_logic_vector(31 downto 0);
		
		signal datak_ch0 : std_logic_vector(3 downto 0);
		signal datak_ch1 : std_logic_vector(3 downto 0);
		signal datak_ch2 : std_logic_vector(3 downto 0);
		signal datak_ch3 : std_logic_vector(3 downto 0);
		
		signal rx_errdetect_ch0_rx_errdetect : std_logic_vector(3 downto 0);
		signal rx_errdetect_ch1_rx_errdetect : std_logic_vector(3 downto 0);
		signal rx_errdetect_ch2_rx_errdetect : std_logic_vector(3 downto 0);
		signal rx_errdetect_ch3_rx_errdetect : std_logic_vector(3 downto 0);
		
      signal rx_disperr_ch0_rx_disperr : std_logic_vector(3 downto 0);
		signal rx_disperr_ch1_rx_disperr : std_logic_vector(3 downto 0);
		signal rx_disperr_ch2_rx_disperr : std_logic_vector(3 downto 0);
		signal rx_disperr_ch3_rx_disperr : std_logic_vector(3 downto 0);
		
		signal enapatternalign_ch0 : std_logic;
		signal enapatternalign_ch1 : std_logic;
		signal enapatternalign_ch2 : std_logic;
		signal enapatternalign_ch3 : std_logic;
				
		signal push_old : std_logic := '0';
		signal rand_num_data : std_logic_vector(31 downto 0);
		signal rand_num_valid : std_logic;
		signal compare_random_ch0 : std_logic;
				
		--- linear shift reg ---
		signal old_shift : std_logic_vector(7 downto 0) := "00100100";
		signal word_counter : std_logic_vector(127 downto 0);
		signal error_counter : std_logic_vector(127 downto 0);
		signal en_shift : std_logic;
		signal n_0 : std_logic_vector(7 downto 0);
		signal n_1 : std_logic_vector(7 downto 0);
		signal n_2 : std_logic_vector(7 downto 0);
		signal n_3 : std_logic_vector(7 downto 0);
		signal n_local : std_logic_vector(7 downto 0);
		
		--- debouncer ---
		signal push_button0_db : std_logic;
		signal push_button1_db : std_logic;
		signal push_button2_db : std_logic;
		signal push_button3_db : std_logic;
		
		signal push_button0_db_156 : std_logic;
		signal push_button1_db_156 : std_logic;
		signal push_button2_db_156 : std_logic;
		signal push_button3_db_156 : std_logic;
		
		-- data generartor stuff
		signal event_counter : std_logic_vector(31 downto 0);
		signal time_counter : std_logic_vector(63 downto 0);

		-- data generartor64 stuff
		signal event_counter64 : std_logic_vector(31 downto 0);
		signal time_counter64 : std_logic_vector(63 downto 0);

		-- second data generartor64 stuff
		signal event2_counter64 : std_logic_vector(31 downto 0);
		signal time2_counter64 : std_logic_vector(63 downto 0);

		--- Test stuff ---
		signal counter_test : std_logic_vector(63 downto 0);
		signal counter_flag : std_logic;
		
		--- SPI stuff ---
      signal spi_clk : std_logic;
		signal spi_ss_n : std_logic;
		
		--- trigger ---
		--signal trigger : std_logic;
		
		--- sorting ---
		signal data_out   : std_logic_vector(63 downto 0);
		signal clk_fast 	: std_logic; -- 312 MHZ
		signal error      : std_logic;
			
		
		--- running stuff ---
		signal enable_sig : std_logic;
		signal random_seed_sig : std_logic_vector(7 downto 0);
		signal sync_reset_sig : std_logic;
		signal data_en_sig : std_logic;
		signal data_out_sig : std_logic_vector(31 downto 0);
		signal fifo_data_out : std_logic_vector(31 downto 0);
		signal fifo_data_full : std_logic;
		signal fifo_data_empty : std_logic;
		signal fifo_rdreq : std_logic;
		signal pixel_data_ready_sig : std_logic;
		signal pixel_data_out_sig : std_logic_vector(31 downto 0);
		
		signal counter_256 	: std_logic_vector(255 downto 0);
		
		signal rdreg_fifo_dma : std_logic;
		signal rdempty_fifo_0 : std_logic;
		signal rdempty_fifo_1 : std_logic;
		signal rdempty_fifo_2 : std_logic;
		signal rdempty_fifo_3 : std_logic;
		
		signal dma_data_ch0 : std_logic_vector(31 downto 0);
		signal dma_data_ch1 : std_logic_vector(31 downto 0);
		signal dma_data_ch2 : std_logic_vector(31 downto 0);
		signal dma_data_ch3 : std_logic_vector(31 downto 0);
		
		signal fiforclk	  : std_logic;
		signal ch0_align_status : std_logic;
		signal ch1_align_status : std_logic;
		signal ch2_align_status : std_logic;
		signal ch3_align_status : std_logic;
		
		signal clk_125 : std_logic;
		
		signal flash_ce_n_i : std_logic;
		signal cpu_reset_n_q : std_logic;
		signal i2c_scl_in   : std_logic;
		signal i2c_scl_oe   : std_logic;
		signal i2c_sda_in   : std_logic;
		signal i2c_sda_oe   : std_logic;
		
		-- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
		signal ZERO : std_logic := '0';
		attribute keep : boolean;
		attribute keep of ZERO : signal is true;
		
		signal datak_transceiver_0  : std_logic_vector(3 downto 0);
		signal datak_transceiver_1  : std_logic_vector(3 downto 0);
		signal datak_transceiver_2  : std_logic_vector(3 downto 0);
		signal datak_transceiver_3  : std_logic_vector(3 downto 0);
		
		signal data_transceiver_0  : std_logic_vector(31 downto 0);
		signal data_transceiver_1  : std_logic_vector(31 downto 0);
		signal data_transceiver_2  : std_logic_vector(31 downto 0);
		signal data_transceiver_3  : std_logic_vector(31 downto 0);
		
		signal datak_transceiver_0_algin  : std_logic_vector(3 downto 0);
		signal datak_transceiver_1_algin  : std_logic_vector(3 downto 0);
		signal datak_transceiver_2_algin  : std_logic_vector(3 downto 0);
		signal datak_transceiver_3_algin  : std_logic_vector(3 downto 0);
		
		signal data_transceiver_0_algin  : std_logic_vector(31 downto 0);
		signal data_transceiver_1_algin  : std_logic_vector(31 downto 0);
		signal data_transceiver_2_algin  : std_logic_vector(31 downto 0);
		signal data_transceiver_3_algin  : std_logic_vector(31 downto 0);
		
		signal ch0_empty : std_logic;
		signal ch1_empty : std_logic;
		signal ch2_empty : std_logic;
		signal ch3_empty : std_logic;
		
		signal ch0_fifo_out : std_logic_vector(35 downto 0);
		signal ch1_fifo_out : std_logic_vector(35 downto 0);
		signal ch2_fifo_out : std_logic_vector(35 downto 0);
		signal ch3_fifo_out : std_logic_vector(35 downto 0);
		
		signal fifo_read : std_logic;
		
		signal tx_coreclkin_ch0_clk : std_logic;
		signal tx_coreclkin_ch1_clk : std_logic;
		signal tx_coreclkin_ch2_clk : std_logic;
		signal tx_coreclkin_ch3_clk : std_logic;
		signal tx_clkout_ch0_clk    : std_logic;
		signal tx_clkout_ch1_clk    : std_logic;
		signal tx_clkout_ch2_clk    : std_logic;
		signal tx_clkout_ch3_clk    : std_logic;
		
		signal tx_data_ch0	: std_logic_vector(31 downto 0);
		signal tx_data_ch1	: std_logic_vector(31 downto 0);
		signal tx_data_ch2	: std_logic_vector(31 downto 0);
		signal tx_data_ch3	: std_logic_vector(31 downto 0);
		
		signal tx_datak_ch0	: std_logic_vector(3 downto 0);
		signal tx_datak_ch1	: std_logic_vector(3 downto 0);
		signal tx_datak_ch2	: std_logic_vector(3 downto 0);
		signal tx_datak_ch3	: std_logic_vector(3 downto 0);
		
		signal tx_clk_ch0 : std_logic;
		signal tx_clk_ch1 : std_logic;
		signal tx_clk_ch2 : std_logic;
		signal tx_clk_ch3 : std_logic;
		
		signal mem_data_out : std_logic_vector(127 downto 0);
		signal mem_datak_out : std_logic_vector(15 downto 0);
		
		signal sc_wren : std_logic;
		
		signal fifo_data_in_ch0 : std_logic_vector(31 downto 0);
		signal fifo_data_in_ch1 : std_logic_vector(31 downto 0);
		signal fifo_data_in_ch2 : std_logic_vector(31 downto 0);
		signal fifo_data_in_ch3 : std_logic_vector(31 downto 0);
		
		signal fifo_wrreq_ch0 : std_logic;
		signal fifo_wrreq_ch1 : std_logic;
		signal fifo_wrreq_ch2 : std_logic;
		signal fifo_wrreq_ch3 : std_logic;
		
		signal pb_in : std_logic_vector(2 downto 0);
		
		signal flash_tcm_address_out : std_logic_vector(27 downto 0);
		
		signal clks_read : std_logic_vector(4 - 1 downto 0);
		signal clks_write : std_logic_vector(4 - 1 downto 0);
		signal data_in : std_logic_vector(4 * 32 - 1 downto 0);
		signal fpga_id_in : std_logic_vector(4 * 16 - 1 downto 0);
		signal enables_in : std_logic_vector(4 - 1 downto 0);
		signal data_algin : std_logic_vector(63 downto 0);
		
		signal sc_ch0 : std_logic_vector(31 downto 0);
		signal sc_ch1 : std_logic_vector(31 downto 0);
		signal sc_ch2 : std_logic_vector(31 downto 0);
		signal sc_ch3 : std_logic_vector(31 downto 0);
		
		signal sck_ch0 : std_logic_vector(3 downto 0);
		signal sck_ch1 : std_logic_vector(3 downto 0);
		signal sck_ch2 : std_logic_vector(3 downto 0);
		signal sck_ch3 : std_logic_vector(3 downto 0);
		
		signal sc_ready_ch0 : std_logic;
		signal sc_ready_ch1 : std_logic;
		signal sc_ready_ch2 : std_logic;
		signal sc_ready_ch3 : std_logic;
		
		signal fifo_datak_in_ch0 : std_logic_vector(3 downto 0);
		signal fifo_datak_in_ch1 : std_logic_vector(3 downto 0);
		signal fifo_datak_in_ch2 : std_logic_vector(3 downto 0);
		signal fifo_datak_in_ch3 : std_logic_vector(3 downto 0);
		
		signal aligned_ch0 : std_logic;
		signal aligned_ch1 : std_logic;
		signal aligned_ch2 : std_logic;
		signal aligned_ch3 : std_logic;
		
		signal wd_rst_n     : std_logic;
		
		signal cpu_pio_i : std_logic_vector(31 downto 0);
		
		signal flash_rst_n : std_logic;
		
		signal avm_qsfp : work.mu3e.avalon_t;
		
		
begin 

--------- I/O ---------

clk <= CLK_50_B2J;
reset_n <= not reset;
reset <= not push_button0_db;
LED_BRACKET(0) <= aligned_ch0;
LED_BRACKET(1) <= aligned_ch1;
LED_BRACKET(2) <= aligned_ch2;
LED_BRACKET(3) <= aligned_ch3;
--cpu_reset_n_q <= CPU_RESET_n;
LED(1) <= SW(1); --- SW for datak transceiver

--reset <= not push_button0_db; -- for receiver

--LED_BRACKET(0) <= rx_is_lockedtoref(0);
--LED_BRACKET(1) <= rx_ready(0);
--LED_BRACKET(2) <= rx_syncstatus_ch0_rx_syncstatus(0);
--LED_BRACKET(3) <= writeregs(LED_REGISTER_W)(3);

--SMA_CLKOUT <= trigger;

--status_rr(CH_ALIGN) <= ch0_align_status and ch1_align_status and ch2_align_status and ch3_align_status;

--input_clk <= refclk2_qr1_p;

receiver_clk : component ip_clk_ctrl
  port map (
		inclk  => SMA_CLKIN,
		outclk => input_clk
);

--------- Debouncer/seg7 ---------

i_debouncer : entity work.debouncer
 generic map (
	  W => 4,
	  N => 125 * 10**3 -- 1ms
 )
 port map (
	  d 		=> BUTTON,
	  q(0) 	=> push_button0_db,
	  q(1) 	=> push_button1_db,
	  q(2) 	=> push_button2_db,
	  q(3) 	=> push_button3_db,
	  arst_n => cpu_reset_n_q,
	  clk 	=> clk--,
 );

clk_125_cnt_p : process(clk)
begin
	if rising_edge(clk) then
		clk_125_cnt <= clk_125_cnt + 1;
	end if;
end process clk_125_cnt_p;

clk_125_cnt_p_2 : process(input_clk)
begin
	if rising_edge(input_clk) then
		clk_125_cnt2 <= clk_125_cnt2 + 1;
	end if;
end process clk_125_cnt_p_2;

segment0 : component seg7_lut
	port map (
		clk => clk, 
		hex => clk_125_cnt(27 downto 24),
		seg => HEX0_D
);

segment1 : component seg7_lut
	port map (
		clk => input_clk, 
		hex => clk_125_cnt2(27 downto 24),
		seg => HEX1_D
);

------------- NIOS -------------

nios2 : component nios
port map (
	avm_qsfp_address          				=> avm_qsfp.address(13 downto 0),
	avm_qsfp_read             				=> avm_qsfp.read,
	avm_qsfp_readdata         				=> avm_qsfp.readdata,
	avm_qsfp_write            				=> avm_qsfp.write,
	avm_qsfp_writedata        				=> avm_qsfp.writedata,
	avm_qsfp_waitrequest      				=> avm_qsfp.waitrequest,
	avm_sc_address            				=> open,
	avm_sc_read               				=> open,
	avm_sc_readdata           				=> x"00000000",
	avm_sc_write              				=> open,
	avm_sc_writedata          				=> open,
	avm_sc_waitrequest        				=> '0',
	clk_clk                    			=> clk,
	clk_data_clk               			=> rx_clkout_ch0_clk,
	flash_tcm_address_out				 	=> flash_tcm_address_out,
	flash_tcm_data_out 						=> FLASH_D,
	flash_tcm_read_n_out(0) 				=> FLASH_OE_n,
	flash_tcm_write_n_out(0) 				=> FLASH_WE_n,
	flash_tcm_chipselect_n_out(0) 		=> flash_ce_n_i,
	i2c_sda_in                 			=> i2c_sda_in,
	i2c_scl_in                 			=> i2c_scl_in,
	i2c_sda_oe                 			=> i2c_sda_oe,
	i2c_scl_oe                 			=> i2c_scl_oe,
	pio_export									=> cpu_pio_i,
	pio_in_export              			=> rx_parallel_data_ch0_rx_parallel_data,              --   pio_in.export
	pio_out_export             			=> open,             --  pio_out.export
	rst_reset_n                			=> cpu_reset_n_q,
	rst_data_reset_n           			=> not reset,
	spi_MISO                   			=> RS422_DIN,
	spi_MOSI                   			=> RS422_DOUT,
	spi_SCLK                   			=> RJ45_LED_R,
	spi_SS_n                   			=> RS422_DE
);

-- generate reset sequence for flash and cpu
reset_ctrl_i : entity work.reset_ctrl
generic map (
		W => 2,
		N => 125 * 10**5 -- 100ms
	)
port map (
		rstout_n(1) => flash_rst_n,
		rstout_n(0) => cpu_reset_n_q,
		rst_n => CPU_RESET_n and wd_rst_n,
		clk => clk--,
);

watchdog_i : entity work.watchdog
generic map (
		W => 4,
		N => 125 * 10**6 -- 1s
)
port map (
		d => cpu_pio_i(3 downto 0),

		rstout_n => wd_rst_n,

		rst_n => CPU_RESET_n,
		clk => clk--,
);

FLASH_A <= flash_tcm_address_out(27 downto 2);

FLASH_CE_n <= (flash_ce_n_i, flash_ce_n_i);
FLASH_ADV_n <= '0';
FLASH_CLK <= '0';
FLASH_RESET_n <= flash_rst_n;

i2c_scl_in <= not i2c_scl_oe;
FAN_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';
TEMP_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';
POWER_MONITOR_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';

i2c_sda_in <= FAN_I2C_SDA and
				TEMP_I2C_SDA and
				POWER_MONITOR_I2C_SDA and
				'1';
FAN_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';
TEMP_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';
POWER_MONITOR_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';

------------- Receiving Data and word aligning -------------

u0 : component receiver_switching
        port map (
            clk_qsfp_clk                            	=> input_clk,
				reset_1_reset                           	=> reset,
				
				rx_bitslip_ch0_rx_bitslip               	=> open,
				rx_bitslip_ch1_rx_bitslip         		 	=> open,
				rx_bitslip_ch2_rx_bitslip         		 	=> open,
				rx_bitslip_ch3_rx_bitslip         		 	=> open,
				rx_cdr_refclk0_clk                      	=> input_clk,
				rx_clkout_ch0_clk 								=> rx_clkout_ch0_clk,
				rx_clkout_ch1_clk									=> rx_clkout_ch1_clk,  
				rx_clkout_ch2_clk 								=> rx_clkout_ch2_clk,
				rx_clkout_ch3_clk 								=> rx_clkout_ch3_clk,  
				rx_coreclkin_ch0_clk 							=> rx_clkout_ch0_clk,
				rx_coreclkin_ch1_clk								=> rx_clkout_ch1_clk, 
				rx_coreclkin_ch2_clk								=> rx_clkout_ch2_clk,
				rx_coreclkin_ch3_clk								=> rx_clkout_ch3_clk,
				rx_datak_ch0_rx_datak          	 		 	=> rx_datak_ch0_rx_datak,
				rx_datak_ch1_rx_datak            		 	=> rx_datak_ch1_rx_datak,
				rx_datak_ch2_rx_datak                     => rx_datak_ch2_rx_datak,
				rx_datak_ch3_rx_datak 							=> rx_datak_ch3_rx_datak,
				rx_is_lockedtoref_ch0_rx_is_lockedtoref 	=> rx_is_lockedtoref(0),
				rx_is_lockedtoref_ch1_rx_is_lockedtoref	=> rx_is_lockedtoref(1),
				rx_is_lockedtoref_ch2_rx_is_lockedtoref 	=> rx_is_lockedtoref(2),
				rx_is_lockedtoref_ch3_rx_is_lockedtoref	=> rx_is_lockedtoref(3),
				rx_parallel_data_ch0_rx_parallel_data   	=> rx_parallel_data_ch0_rx_parallel_data,
				rx_parallel_data_ch1_rx_parallel_data   	=> rx_parallel_data_ch1_rx_parallel_data,
				rx_parallel_data_ch2_rx_parallel_data   	=> rx_parallel_data_ch2_rx_parallel_data,
				rx_parallel_data_ch3_rx_parallel_data   	=> rx_parallel_data_ch3_rx_parallel_data,
				rx_patterndetect_ch0_rx_patterndetect   	=> rx_patterndetect_ch0_rx_patterndetect,
				rx_patterndetect_ch1_rx_patterndetect   	=> rx_patterndetect_ch1_rx_patterndetect,
				rx_patterndetect_ch2_rx_patterndetect   	=> rx_patterndetect_ch2_rx_patterndetect,
				rx_patterndetect_ch3_rx_patterndetect   	=> rx_patterndetect_ch3_rx_patterndetect,
				rx_serial_data_ch0_rx_serial_data       	=> QSFPA_RX_p(0),
				rx_serial_data_ch1_rx_serial_data       	=> QSFPA_RX_p(1),
				rx_serial_data_ch2_rx_serial_data       	=> QSFPA_RX_p(2),
				rx_serial_data_ch3_rx_serial_data       	=> QSFPA_RX_p(3),
				rx_seriallpbken_ch0_rx_seriallpbken     	=> '0',
				rx_seriallpbken_ch1_rx_seriallpbken     	=> '0',
				rx_seriallpbken_ch2_rx_seriallpbken     	=> '0',
				rx_seriallpbken_ch3_rx_seriallpbken     	=> '0',
				rx_syncstatus_ch0_rx_syncstatus  		 	=> rx_syncstatus_ch0_rx_syncstatus,
				rx_syncstatus_ch1_rx_syncstatus  		 	=> rx_syncstatus_ch1_rx_syncstatus,
				rx_syncstatus_ch2_rx_syncstatus  		 	=> rx_syncstatus_ch2_rx_syncstatus,
				rx_syncstatus_ch3_rx_syncstatus  		 	=> rx_syncstatus_ch3_rx_syncstatus,
				rx_ready0_rx_ready								=> rx_ready(0),
				rx_ready1_rx_ready								=> rx_ready(1),
				rx_ready2_rx_ready								=> rx_ready(2),
				rx_ready3_rx_ready								=> rx_ready(3),
				rx_errdetect_ch0_rx_errdetect           	=> rx_errdetect_ch0_rx_errdetect,
				rx_errdetect_ch1_rx_errdetect           	=> rx_errdetect_ch1_rx_errdetect,
				rx_errdetect_ch2_rx_errdetect           	=> rx_errdetect_ch2_rx_errdetect,
				rx_errdetect_ch3_rx_errdetect           	=> rx_errdetect_ch3_rx_errdetect,
				rx_disperr_ch0_rx_disperr               	=> rx_disperr_ch0_rx_disperr,
				rx_disperr_ch1_rx_disperr               	=> rx_disperr_ch1_rx_disperr,
				rx_disperr_ch2_rx_disperr               	=> rx_disperr_ch2_rx_disperr,
				rx_disperr_ch3_rx_disperr               	=> rx_disperr_ch3_rx_disperr,
				
				tx_parallel_data_unused_tx_parallel_data        => (others => '0'),
            tx_parallel_data_ch3_tx_parallel_data           => tx_data_ch3,
            tx_parallel_data_ch2_tx_parallel_data           => tx_data_ch2,
            tx_parallel_data_ch1_tx_parallel_data           => tx_data_ch1,
            tx_parallel_data_ch0_tx_parallel_data           => tx_data_ch0,
            tx_clkout_ch3_clk                               => tx_clk_ch3,
            tx_clkout_ch2_clk                               => tx_clk_ch2,
            tx_clkout_ch1_clk                               => tx_clk_ch1,
            tx_clkout_ch0_clk                               => tx_clk_ch0,
            tx_coreclkin_ch3_clk                            => tx_clk_ch3,
            tx_coreclkin_ch2_clk                            => tx_clk_ch2,
            tx_coreclkin_ch1_clk                            => tx_clk_ch1,
            tx_coreclkin_ch0_clk                            => tx_clk_ch0,
            tx_serial_data_ch3_tx_serial_data               => QSFPA_TX_p(3),
            tx_serial_data_ch2_tx_serial_data               => QSFPA_TX_p(2),
            tx_serial_data_ch1_tx_serial_data               => QSFPA_TX_p(1),
            tx_serial_data_ch0_tx_serial_data               => QSFPA_TX_p(0),
				tx_datak_ch0_tx_datak               				=> tx_datak_ch0,
				tx_datak_ch1_tx_datak               				=> tx_datak_ch1,
				tx_datak_ch2_tx_datak               				=> tx_datak_ch2,
				tx_datak_ch3_tx_datak               				=> tx_datak_ch3  
);

word_align_ch0 : component rx_align
    generic map (
        Nb 						=> 4,
		  K 						=> X"BC"
    )
    port map (
        data    				=> data_ch0,
        datak   				=> datak_ch0,
        lock    				=> aligned_ch0,
        datain  				=> rx_parallel_data_ch0_rx_parallel_data,
        datakin 				=> rx_datak_ch0_rx_datak,
        syncstatus 			=> rx_syncstatus_ch0_rx_syncstatus,
        patterndetect 		=> rx_patterndetect_ch0_rx_patterndetect,
        enapatternalign 	=> enapatternalign_ch0,
        errdetect   			=> rx_errdetect_ch0_rx_errdetect,
        disperr     			=> rx_disperr_ch0_rx_disperr,
        rst_n   				=> reset_n,
        clk     				=> rx_clkout_ch0_clk
    );
	 
word_align_ch1 : component rx_align
    generic map (
        Nb 						=> 4,
		  K 						=> X"BC"
    )
    port map (
        data    				=> data_ch1,
        datak   				=> datak_ch1,
        lock    				=> aligned_ch1,
		  datain  				=> rx_parallel_data_ch1_rx_parallel_data,
        datakin 				=> rx_datak_ch1_rx_datak,
        syncstatus 			=> rx_syncstatus_ch1_rx_syncstatus,
        patterndetect 		=> rx_patterndetect_ch1_rx_patterndetect,
        enapatternalign 	=> enapatternalign_ch1,
        errdetect   			=> rx_errdetect_ch1_rx_errdetect,
        disperr     			=> rx_disperr_ch1_rx_disperr,
        rst_n   				=> reset_n,
        clk     				=> rx_clkout_ch1_clk
    );
	 
word_align_ch2 : component rx_align
    generic map (
        Nb 						=> 4,
		  K 						=> X"BC"      
    )
    port map (
        data    				=> data_ch2,
        datak   				=> datak_ch2,
        lock    				=> aligned_ch2,
        datain  				=> rx_parallel_data_ch2_rx_parallel_data,
        datakin 				=> rx_datak_ch2_rx_datak,
        syncstatus 			=> rx_syncstatus_ch2_rx_syncstatus,
        patterndetect 		=> rx_patterndetect_ch2_rx_patterndetect,
        enapatternalign 	=> enapatternalign_ch2,
        errdetect   			=> rx_errdetect_ch2_rx_errdetect,
        disperr     			=> rx_disperr_ch2_rx_disperr,
        rst_n   				=> reset_n,
        clk     				=> rx_clkout_ch2_clk
    );
	 
word_align_ch3 : component rx_align
    generic map (
        Nb 						=> 4,
		  K 						=> X"BC"
    )
    port map (
        data    				=> data_ch3,
        datak   				=> datak_ch3,
        lock    				=> aligned_ch3,
        datain  				=> rx_parallel_data_ch3_rx_parallel_data,
        datakin 				=> rx_datak_ch3_rx_datak,
        syncstatus 			=> rx_syncstatus_ch3_rx_syncstatus,
        patterndetect 		=> rx_patterndetect_ch3_rx_patterndetect,
        enapatternalign 	=> enapatternalign_ch3,
        errdetect   			=> rx_errdetect_ch3_rx_errdetect,
        disperr     			=> rx_disperr_ch3_rx_disperr,
        rst_n   				=> reset_n,
        clk     				=> rx_clkout_ch3_clk
    );
	 
------------- data demerger -------------

data_demerger0 : component data_demerge
    port map(
		clk				=> rx_clkout_ch0_clk, 	-- receive clock (156.25 MHz)
 		reset				=> not reset_n,
		aligned			=> aligned_ch0,			-- word alignment achieved
		data_in			=>	data_ch0,				-- optical from frontend board
		datak_in			=> datak_ch0,
		data_out			=> fifo_data_in_ch0,		-- to sorting fifos
		data_ready		=>	fifo_wrreq_ch0,	  	-- write req for sorting fifos
		datak_out      => fifo_datak_in_ch0,
		sc_out			=> sc_ch0,							-- slowcontrol from frontend board
		sc_out_ready	=> sc_ready_ch0,
		fpga_id			=> open,						-- FPGA ID of the connected frontend board
		sck_out      	=> sck_ch0
);

data_demerger1 : component data_demerge
    port map(
		clk				=> rx_clkout_ch1_clk, 	-- receive clock (156.25 MHz)
 		reset				=> not reset_n,
		aligned			=> aligned_ch1,			-- word alignment achieved
		data_in			=>	data_ch1,				-- optical from frontend board
		datak_in			=> datak_ch1,
		data_out			=> fifo_data_in_ch1,		-- to sorting fifos
		datak_out      => fifo_datak_in_ch1,
		data_ready		=>	fifo_wrreq_ch1,	  	-- write req for sorting fifos	
		sc_out			=> sc_ch1,							-- slowcontrol from frontend board
		sc_out_ready	=> sc_ready_ch1,
		fpga_id			=> open,						-- FPGA ID of the connected frontend board
		sck_out      	=> sck_ch1
);

data_demerger2 : component data_demerge
    port map(
		clk				=> rx_clkout_ch2_clk, 	-- receive clock (156.25 MHz)
 		reset				=> not reset_n,
		aligned			=> aligned_ch2,			-- word alignment achieved
		data_in			=>	data_ch2,				-- optical from frontend board
		datak_in			=> datak_ch2,
		data_out			=> fifo_data_in_ch2,		-- to sorting fifos
		datak_out      => fifo_datak_in_ch2,
		data_ready		=>	fifo_wrreq_ch2,	  	-- write req for sorting fifos	
		sc_out			=> sc_ch2,							-- slowcontrol from frontend board
		sc_out_ready	=> sc_ready_ch2,
		fpga_id			=> open,						-- FPGA ID of the connected frontend board
		sck_out      	=> sck_ch2
);

data_demerger3 : component data_demerge
    port map(
		clk				=> rx_clkout_ch3_clk, 	-- receive clock (156.25 MHz)
 		reset				=> not reset_n,
		aligned			=> aligned_ch3,			-- word alignment achieved
		data_in			=>	data_ch3,				-- optical from frontend board
		datak_in			=> datak_ch3,
		data_out			=> fifo_data_in_ch3,		-- to sorting fifos
		datak_out      => fifo_datak_in_ch3,
		data_ready		=>	fifo_wrreq_ch3,	  	-- write req for sorting fifos	
		sc_out			=> sc_ch3,							-- slowcontrol from frontend board
		sc_out_ready	=> sc_ready_ch3,
		fpga_id			=> open,						-- FPGA ID of the connected frontend board
		sck_out      	=> sck_ch3
);
	 
	 
------------- time algining data -------------

--pll_algining : component ip_pll_312
--        port map (
--            rst      => reset,      --   reset.reset
--            refclk   => input_clk,   --  refclk.clk
--            locked   => open,   --  locked.export
--            outclk_0 => clk_fast  -- outclk0.clk
--    );
--
--algining_data : sw_algin_data
--generic map(
--	NLINKS => 4
--)
--port map(
--	clks_read         	 => clks_read, -- 156,25 MHZ
--	clks_write			    => clks_write, -- 312,50 MHZ
--
--	clk_node_write      	 => clk,--: in  std_logic; -- 156,25 MHZ
--	clk_node_read     	 => clk,--: in  std_logic; -- To be defined
--
--	reset_n					 => reset_n,--: in  std_logic;
--	
--	data_in					 => data_in,
--	fpga_id_in			    => fpga_id_in, -- FPGA-ID
--	
--	enables_in				 => enables_in,
--	
--	node_rdreq				 => '1',
--	
--	data_out					 => data_algin,
--	state_out				 => open,
--	node_full_out			 => open,
--	node_empty_out			 => open
--);
--	
--clks_read <= clk & clk & clk & clk;
--clks_write <= clk & clk & clk & clk;
--data_in <= data_ch0 & data_ch1 & data_ch2 & data_ch3;
--fpga_id_in <= "0000000000000001" & "0000000000000011" & "0000000000000111" & "0000000000001111";
--enables_in <= datak_ch0(0) & datak_ch1(0) & datak_ch2(0) & datak_ch3(0);

------------- transceiver_switching -------------

fifo_read <= (not ch0_empty) and (not ch1_empty) and (not ch2_empty) and (not ch3_empty);

ch0 : component transceiver_fifo -- pixel data / sc -- 
  port map (
		data    => fifo_data_in_ch0 & fifo_datak_in_ch0,    --  fifo_input.datain
		wrreq   => fifo_wrreq_ch0,   --            .wrreq
		rdreq   => fifo_read,   --            .rdreq
		wrclk   => rx_clkout_ch0_clk,   --            .wrclk
		rdclk   => pcie_fastclk_out,--tx_clkout_ch0_clk,   --            .rdclk
		aclr    => reset_n,    --            .aclr
		q       => ch0_fifo_out,       -- fifo_output.dataout
		rdempty => ch0_empty, --            .rdempty
		wrfull  => open   --            .wrfull
  );
  
ch1 : component transceiver_fifo -- pixel data / sc -- 
  port map (
		data    => fifo_data_in_ch1 & fifo_datak_in_ch1,    --  fifo_input.datain
		wrreq   => fifo_wrreq_ch1,   --            .wrreq
		rdreq   => fifo_read,   --            .rdreq
		wrclk   => rx_clkout_ch1_clk,   --            .wrclk
		rdclk   => pcie_fastclk_out,--tx_clkout_ch1_clk,   --            .rdclk
		aclr    => reset_n,    --            .aclr
		q       => ch1_fifo_out,       -- fifo_output.dataout
		rdempty => ch1_empty, --            .rdempty
		wrfull  => open   --            .wrfull
  );
  
ch2 : component transceiver_fifo
  port map (
		data    => fifo_data_in_ch2 & fifo_datak_in_ch2,    --  fifo_input.datain
		wrreq   => fifo_wrreq_ch2,   --            .wrreq
		rdreq   => fifo_read,   --            .rdreq
		wrclk   => rx_clkout_ch2_clk,   --            .wrclk
		rdclk   => pcie_fastclk_out,--tx_clkout_ch2_clk,   --            .rdclk
		aclr    => reset_n,    --            .aclr
		q       => ch2_fifo_out,       -- fifo_output.dataout
		rdempty => ch2_empty, --            .rdempty
		wrfull  => open   --            .wrfull
  );
  
ch3 : component transceiver_fifo
  port map (
		data    => fifo_data_in_ch3 & fifo_datak_in_ch3,    --  fifo_input.datain
		wrreq   => fifo_wrreq_ch3,   --            .wrreq
		rdreq   => fifo_read,   --            .rdreq
		wrclk   => rx_clkout_ch3_clk,   --            .wrclk
		rdclk   => pcie_fastclk_out,--tx_clkout_ch3_clk,   --            .rdclk
		aclr    => reset_n,    --            .aclr
		q       => ch3_fifo_out,       -- fifo_output.dataout
		rdempty => ch3_empty, --            .rdempty
		wrfull  => open   --            .wrfull
  );

tra_switching : component transceiver_switching
	port map (
		clk_qsfp_clk                          => input_clk,                 
		pll_refclk0_clk                       => input_clk,                    
		reset_1_reset                         => reset,                     
		tx_serial_data_ch0_tx_serial_data     => QSFPB_TX_p(0),     
		tx_serial_data_ch1_tx_serial_data     => QSFPB_TX_p(1),   
		tx_serial_data_ch2_tx_serial_data     => QSFPB_TX_p(2),     
		tx_serial_data_ch3_tx_serial_data     => QSFPB_TX_p(3),      		
		tx_datak_ch0_tx_datak                 => "0001",
		tx_datak_ch1_tx_datak                 => "0001",
		tx_datak_ch2_tx_datak                 => "0001",
		tx_datak_ch3_tx_datak                 => "0001",
		tx_parallel_data_ch0_tx_parallel_data => x"0FFECABC",
		tx_parallel_data_ch1_tx_parallel_data => x"1FFECABC",
		tx_parallel_data_ch2_tx_parallel_data => x"2FFECABC",
		tx_parallel_data_ch3_tx_parallel_data => x"3FFECABC",
		tx_coreclkin_ch0_clk                  => tx_clkout_ch0_clk,
		tx_coreclkin_ch1_clk                  => tx_clkout_ch1_clk,
		tx_coreclkin_ch2_clk                  => tx_clkout_ch2_clk,
		tx_coreclkin_ch3_clk                  => tx_clkout_ch3_clk,
		tx_clkout_ch0_clk                     => tx_clkout_ch0_clk,
		tx_clkout_ch1_clk                     => tx_clkout_ch1_clk,
		tx_clkout_ch2_clk                     => tx_clkout_ch2_clk,
		tx_clkout_ch3_clk                     => tx_clkout_ch3_clk
);

------------- Slow Control -------------

sc_master_ch0:sc_master
	generic map(
		NLINKS => 4
	)
	port map(
		clk					=> tx_clk_ch0,
		reset_n				=> push_button0_db,
		enable				=> '1',
		mem_data_in			=> writememreaddata,
		mem_addr				=> writememreadaddr,
		mem_data_out		=> mem_data_out,
		mem_data_out_k		=> mem_datak_out,
		done					=> open,
		stateout				=> open
);

sc_slave_ch0:sc_slave
	port map(
		clk					=> tx_clk_ch0,--rx_clkout_ch0_clk,
		reset_n				=> push_button0_db,
		enable				=> '1',
		link_data_in		=> mem_data_out(31 downto 0),--sc_ch0,--data_ch0,
		link_data_in_k		=> mem_datak_out(3 downto 0),--sck_ch0,--datak_ch0,
		mem_addr_out		=> readmem_writeaddr(15 downto 0),
		mem_data_out		=> readmem_writedata,
		mem_wren				=> readmem_wren,
		done					=> open,
		stateout				=> open
);

tx_data_ch0 <= mem_data_out(31 downto 0);
tx_datak_ch0 <= mem_datak_out(3 downto 0);

--- SC transceiver ---
sck_process_1 : process(tx_clk_ch1)
begin
if rising_edge(tx_clk_ch1) then
	if (SW(1) = '0') then
		tx_datak_ch1 <= "0001";
		tx_data_ch1 <= x"000000BC";
	elsif (SW(1) = '1') then
		tx_datak_ch1 <= "0000";
		tx_data_ch1 <= x"AFFECAFE";
   end if;
end if;
end process;

sck_process_2 : process(tx_clk_ch2)
begin
if rising_edge(tx_clk_ch2) then
	if (SW(1) = '0') then
		tx_datak_ch2 <= "0001";
		tx_data_ch2 <= x"000000BC";
	elsif (SW(1) = '1') then
		tx_datak_ch2 <= "0000";
		tx_data_ch2 <= x"AFFECAFE";
   end if;
end if;
end process;

sck_process_3 : process(tx_clk_ch3)
begin
if rising_edge(tx_clk_ch3) then
	if (SW(1) = '0') then
		tx_datak_ch3 <= "0001";
		tx_data_ch3 <= x"000000BC";
	elsif (SW(1) = '1') then
		tx_datak_ch3 <= "0000";
		tx_data_ch3 <= x"AFFECAFE";
   end if;
end if;
end process;

------------- PCIe -------------

resetlogic:reset_logic
	port map(
		clk                     => clk,--clk_125,
		rst_n                   => push_button0_db,

		reset_register          => writeregs(RESET_REGISTER_W),
		reset_reg_written       => regwritten(RESET_REGISTER_W),

		resets                  => resets,
		resets_n                => resets_n                                                             
);

vreg:version_reg
	port map(
		data_out  => readregs_slow(VERSION_REGISTER_R)(27 downto 0)
);

--Sync read regs from slow  (50 MHz) to fast (250 MHz) clock
process(pcie_fastclk_out)
begin
	if(pcie_fastclk_out'event and pcie_fastclk_out = '1') then
		clk_sync <= clk;
		clk_last <= clk_sync;
		
		if(clk_sync = '1' and clk_last = '0') then
			readregs(PLL_REGISTER_R) 			<= readregs_slow(PLL_REGISTER_R);
			readregs(VERSION_REGISTER_R) 		<= readregs_slow(VERSION_REGISTER_R);
		end if;
		readregs(EVENTCOUNTER_REGISTER_R)		<= event_counter;
		readregs(EVENTCOUNTER64_REGISTER_R)		<= event_counter64;
		readregs(TIMECOUNTER_LOW_REGISTER_R)	<= time_counter(31 downto 0);
		readregs(TIMECOUNTER_HIGH_REGISTER_R)	<= time_counter(63 downto 32);
		readregs(EVENT2COUNTER64_REGISTER_R)	<= event2_counter64;
	end if;
end process;

-- Increase address
--process(pcie_fastclk_out, resets_n(RESET_BIT_DATAGEN))
--begin
--	if(resets_n(RESET_BIT_DATAGEN) = '0') then
--		readmem_writeaddr  <= (others => '0');
--	elsif(pcie_fastclk_out'event and pcie_fastclk_out = '1') then
--		if(readmem_wren = '1') then
--			readmem_writeaddr    <= readmem_writeaddr + '1';
--			readregs(MEM_WRITEADDR_LOW_REGISTER_R) <= readmem_writeaddr(31 downto 0);
--			readregs(MEM_WRITEADDR_HIGH_REGISTER_R) <= readmem_writeaddr(63 downto 32);
--		end if;
--	end if;
--end process;

--Prolong regwritten signals for 50 MHz clock
process(pcie_fastclk_out)
begin
	if(pcie_fastclk_out'event and pcie_fastclk_out = '1') then
		regwritten_del1 <= regwritten_fast;
		regwritten_del2 <= regwritten_del1;
		regwritten_del3 <= regwritten_del2;
		regwritten_del4 <= regwritten_del3;
		for I in 63 downto 0 loop
			if(regwritten_fast(I) = '1' or regwritten_del1(I) = '1'  or regwritten_del2(I) = '1'  or regwritten_del3(I) = '1'  or regwritten_del4(I) = '1') then
				regwritten(I) <= '1';
			else
			regwritten(I) <= '0';
			end if;
		end loop;
	end if;
end process;


readmem_writeaddr_lowbits <= readmem_writeaddr(15 downto 0);
dmamem_wren <= writeregs(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE);-- and rdreg_fifo_dma;
pb_in <= push_button0_db & push_button1_db & push_button2_db;

pcie_b: pcie_block 
	generic map(
		DMAMEMWRITEADDRSIZE => 11,
		DMAMEMREADADDRSIZE  => 11,
		DMAMEMWRITEWIDTH	  => 256
	)
	port map(
		local_rstn				=> '1',
		appl_rstn				=> '1', --resets_n(RESET_BIT_PCIE),
		refclk					=> PCIE_REFCLK_p,
		pcie_fastclk_out		=> pcie_fastclk_out,
		
		--//PCI-Express--------------------------//25 pins //--------------------------
		pcie_rx_p				=> PCIE_RX_p,
		pcie_tx_p 				=> PCIE_TX_p,
		pcie_refclk_p			=> PCIE_REFCLK_p,
		pcie_led_g2				=> open,
		pcie_led_x1				=> open,
		pcie_led_x4				=> open,
		pcie_led_x8				=> open,
		pcie_perstn 			=> PCIE_PERST_n,
		pcie_smbclk				=> PCIE_SMBCLK,
		pcie_smbdat				=> PCIE_SMBDAT,
		pcie_waken				=> PCIE_WAKE_n,

		-- LEDs
		alive_led		      => open,
		comp_led			      => open,
		L0_led			      => open,

		-- pcie registers (write / read register ,  readonly, read write , in tools/dmatest/rw) -Sync read regs 
		writeregs		      => writeregs,
		regwritten		      => regwritten_fast,
		readregs			      => readregs,

		-- pcie writeable memory
		writememclk		      => tx_clk_ch0,--input_clk,
		writememreadaddr     => writememreadaddr,
		writememreaddata     => writememreaddata,

		-- pcie readable memory
		readmem_data 			=> readmem_writedata,
		readmem_addr 			=> readmem_writeaddr_lowbits,
		readmemclk				=> tx_clk_ch0,--rx_clkout_ch0_clk,
		readmem_wren			=> readmem_wren,
		readmem_endofevent	=> readmem_endofevent,

		-- dma memory 
		dma_data 				=> ch0_fifo_out(35 downto 4) & X"DECAFBAD" & ch1_fifo_out(35 downto 4) & X"DECAFBAD" & ch2_fifo_out(35 downto 4) & X"DECAFBAD" & ch3_fifo_out(35 downto 4) & X"DECAFBAD",--counter_256,--rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data,
		dmamemclk				=> pcie_fastclk_out,--rx_clkout_ch0_clk,--rx_clkout_ch0_clk,
		dmamem_wren				=> dmamem_wren,--'1',
		dmamem_endofevent		=> dmamem_endofevent,
		dmamemhalffull			=> open,--dmamemhalffull,

		-- dma memory
		dma2_data 				=> dma2mem_writedata,
		dma2memclk				=> pcie_fastclk_out,
		dma2mem_wren			=> dma2mem_wren,
		dma2mem_endofevent	=> dma2mem_endofevent,
		dma2memhalffull		=> dma2memhalffull,

		-- test ports  
		testout					=> pcie_testout,
		testout_ena				=> open,
		pb_in						=> pb_in,
		inaddr32_r				=> readregs(inaddr32_r),
		inaddr32_w				=> readregs(inaddr32_w)
);

end;
