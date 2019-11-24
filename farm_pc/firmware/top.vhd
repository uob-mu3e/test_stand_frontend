library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.pcie_components.all;
use work.mudaq_registers.all;
use work.mudaq_components.all;


entity top is
port (
    BUTTON              : in    std_logic_vector(3 downto 0);

    HEX0_D              : out   std_logic_vector(6 downto 0);
--    HEX0_DP             : out   std_logic;

    HEX1_D              : out   std_logic_vector(6 downto 0);
--    HEX1_DP             : out   std_logic;

    LED                 : out   std_logic_vector(3 downto 0) := "0000";
    LED_BRACKET         : out   std_logic_vector(3 downto 0) := "0000";

    SMA_CLKOUT          : out std_logic;
    SMA_CLKIN           : in std_logic;

    RS422_DE            : out   std_logic;
    RS422_DIN           : in    std_logic; -- 1.8-V
    RS422_DOUT          : out   std_logic;
--    RS422_RE_n          : out   std_logic;
--    RJ45_LED_L          : out   std_logic;
    RJ45_LED_R          : out   std_logic;

--    refclk2_qr1_p       : in    std_logic; -- 1.5-V PCML, default 125MHz
--    refclk1_qr0_p       : in    std_logic; -- 1.5-V PCML, default 156.25MHz

    -- //////// FAN ////////
    FAN_I2C_SCL         : out   std_logic;
    FAN_I2C_SDA         : inout std_logic;

    -- //////// FLASH ////////
    FLASH_A             : out   std_logic_vector(26 downto 1);
    FLASH_D             : inout std_logic_vector(31 downto 0);
    FLASH_OE_n          : inout std_logic;
    FLASH_WE_n          : out   std_logic;
    FLASH_CE_n          : out   std_logic_vector(1 downto 0);
    FLASH_ADV_n         : out   std_logic;
    FLASH_CLK           : out   std_logic;
    FLASH_RESET_n       : out   std_logic;

    -- //////// POWER ////////
    POWER_MONITOR_I2C_SCL   : out   std_logic;
    POWER_MONITOR_I2C_SDA   : inout std_logic;

    -- //////// TEMP ////////
    TEMP_I2C_SCL        : out   std_logic;
    TEMP_I2C_SDA        : inout std_logic;

    SW : in std_logic_vector(1 downto 0);

--    clkin_50_top        : in    std_logic; -- 2.5V, default 50MHz

    -- //////// Transiver ////////
    QSFPA_TX_p          : out   std_logic_vector(3 downto 0);
    QSFPB_TX_p          : out   std_logic_vector(3 downto 0);

    QSFPA_RX_p          : in    std_logic_vector(3 downto 0);
    QSFPB_RX_p          : in    std_logic_vector(3 downto 0);

    QSFPA_REFCLK_p      : in    std_logic;
--    QSFPB_REFCLK_p      : in    std_logic;
    QSFPA_LP_MODE       : out   std_logic;
    QSFPA_MOD_SEL_n     : out   std_logic;
    QSFPA_RST_n         : out   std_logic;



    -- //////// PCIE ////////
    PCIE_PERST_n        : in    std_logic;
    PCIE_REFCLK_p       : in    std_logic;
    PCIE_RX_p           : in    std_logic_vector(7 downto 0);
    PCIE_SMBCLK         : in    std_logic;
    PCIE_SMBDAT         : inout std_logic;
    PCIE_TX_p           : out   std_logic_vector(7 downto 0);
    PCIE_WAKE_n         : out   std_logic;

    CPU_RESET_n         : in    std_logic;
    CLK_50_B2J          : in    std_logic--;
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
		
		--signal reset : std_logic;
		
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
		
		--signal enapatternalign_ch0 : std_logic;
		--signal enapatternalign_ch1 : std_logic;
		--signal enapatternalign_ch2 : std_logic;
		--signal enapatternalign_ch3 : std_logic;
		
		--signal lock_ch0 : std_logic;
		--signal lock_ch1 : std_logic;
		--signal lock_ch2 : std_logic;
		--signal lock_ch3 : std_logic;
				
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
		signal flash_tcm_address_out : std_logic_vector(27 downto 0);

		-- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
		signal ZERO : std_logic := '0';
		attribute keep : boolean;
		attribute keep of ZERO : signal is true;


begin

--------- I/O ---------

clk <= CLK_50_B2J; -- for debouncer

--reset <= not push_button0_db; -- for receiver

LED(0) <= not error;
LED(1) <= not resets_n(RESET_NIOS);

LED_BRACKET(0) <= rx_is_lockedtoref(0);
LED_BRACKET(1) <= rx_ready(0);
LED_BRACKET(2) <= rx_syncstatus_ch0_rx_syncstatus(0);
LED_BRACKET(3) <= writeregs(LED_REGISTER_W)(3);

--SMA_CLKOUT <= trigger;

--status_rr(CH_ALIGN) <= ch0_align_status and ch1_align_status and ch2_align_status and ch3_align_status;

--input_clk <= refclk2_qr1_p;

    -------- NIOS --------

cpu_reset_n_q <= push_button1_db;

    e_nios : work.cmp.nios
    port map (
        flash_tcm_address_out           => flash_tcm_address_out,
        flash_tcm_data_out              => FLASH_D,
        flash_tcm_read_n_out(0)         => FLASH_OE_n,
        flash_tcm_write_n_out(0)        => FLASH_WE_n,
        flash_tcm_chipselect_n_out(0)   => flash_ce_n_i,

        i2c_sda_in                      => i2c_sda_in,
        i2c_scl_in                      => i2c_scl_in,
        i2c_sda_oe                      => i2c_sda_oe,
        i2c_scl_oe                      => i2c_scl_oe,

        rst_reset_n                     => push_button3_db,--resets_n(RESET_NIOS)
        clk_clk                         => clk--,
    );

    FLASH_A <= flash_tcm_address_out(27 downto 2);
    FLASH_CE_n <= (flash_ce_n_i, flash_ce_n_i);
    FLASH_ADV_n <= '0';
    FLASH_CLK <= '0';
    FLASH_RESET_n <= cpu_reset_n_q;

    i2c_scl_in <= not i2c_scl_oe;
    FAN_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';
    TEMP_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';
    POWER_MONITOR_I2C_SCL <= ZERO when i2c_scl_oe = '1' else 'Z';

    i2c_sda_in <=
        FAN_I2C_SDA and
        TEMP_I2C_SDA and
        POWER_MONITOR_I2C_SDA and
        '1';
    FAN_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';
    TEMP_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';
    POWER_MONITOR_I2C_SDA <= ZERO when i2c_sda_oe = '1' else 'Z';

--------- Debouncer/seg7 ---------

    e_deb1 : component debouncer
    port map (
        clk => clk,
        din => BUTTON(0),
        dout => push_button0_db
    );

deb2 : component debouncer
	port map(
		clk => clk, 
		din => BUTTON(1), 
		dout => push_button1_db
);

deb3 : component debouncer
	port map(
		clk => clk, 
		din => BUTTON(2), 
		dout => push_button2_db
);

deb4 : component debouncer
	port map(
		clk => clk, 
		din => BUTTON(3), 
		dout => push_button3_db
);

clk_125_cnt_p : process(clk)
begin
	if rising_edge(clk) then
		clk_125_cnt <= clk_125_cnt + 1;
	end if; -- rising_edge
end process clk_125_cnt_p;

clk_125_cnt_p_2 : process(input_clk)
begin
	if rising_edge(input_clk) then
		clk_125_cnt2 <= clk_125_cnt2 + 1;
	end if; -- rising_edge
end process clk_125_cnt_p_2;

e_segment0 : entity work.hex2seg7
port map (
	i_hex => clk_125_cnt(27 downto 24),
	o_seg => HEX0_D
);

e_segment1 : entity work.hex2seg7
port map (
	i_hex => clk_125_cnt2(27 downto 24),
	o_seg => HEX1_D
);

--------- data receiver ---------

receiver_clk : component ip_clk_ctrl
  port map (
		inclk  => SMA_CLKIN,
		outclk => input_clk
);

rec_switching : component receiver_switching
	port map (
		clk_qsfp_clk                            	=> input_clk,
		reset_1_reset                           	=> resets(RESET_BIT_RECEIVER),--reset,
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
		rx_serial_data_ch0_rx_serial_data       	=> QSFPB_RX_p(0),
		rx_serial_data_ch1_rx_serial_data       	=> QSFPB_RX_p(1),
		rx_serial_data_ch2_rx_serial_data       	=> QSFPB_RX_p(2),
		rx_serial_data_ch3_rx_serial_data       	=> QSFPB_RX_p(3),
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
		rx_disperr_ch3_rx_disperr               	=> rx_disperr_ch3_rx_disperr
);

word_align_ch0 : entity work.rx_align
port map (
	o_data              => data_ch0,
	o_datak             => datak_ch0,
	o_locked            => ch0_align_status,
	i_data              => rx_parallel_data_ch0_rx_parallel_data,
	i_datak             => rx_datak_ch0_rx_datak,
	i_syncstatus        => rx_syncstatus_ch0_rx_syncstatus,
	i_patterndetect     => rx_patterndetect_ch0_rx_patterndetect,
	o_enapatternalign   => open, --enapatternalign_ch0,
	i_errdetect         => rx_errdetect_ch0_rx_errdetect,
	i_disperr           => rx_disperr_ch0_rx_disperr,
	i_reset_n           => resets_n(RESET_BIT_WORDALIGN),--not reset,
	i_clk               => rx_clkout_ch0_clk
);

word_align_ch1 : entity work.rx_align
port map (
	o_data              => data_ch1,
	o_datak             => datak_ch1,
	o_locked            => ch1_align_status,
	i_data              => rx_parallel_data_ch1_rx_parallel_data,
	i_datak             => rx_datak_ch1_rx_datak,
	i_syncstatus        => rx_syncstatus_ch1_rx_syncstatus,
	i_patterndetect     => rx_patterndetect_ch1_rx_patterndetect,
	o_enapatternalign   => open, --enapatternalign_ch1,
	i_errdetect         => rx_errdetect_ch1_rx_errdetect,
	i_disperr           => rx_disperr_ch1_rx_disperr,
	i_reset_n           => resets_n(RESET_BIT_WORDALIGN),--not reset,
	i_clk               => rx_clkout_ch1_clk
);

word_align_ch2 : entity work.rx_align
port map (
	o_data              => data_ch2,
	o_datak             => datak_ch2,
	o_locked            => ch2_align_status,
	i_data              => rx_parallel_data_ch2_rx_parallel_data,
	i_datak             => rx_datak_ch2_rx_datak,
	i_syncstatus        => rx_syncstatus_ch2_rx_syncstatus,
	i_patterndetect     => rx_patterndetect_ch2_rx_patterndetect,
	o_enapatternalign   => open, --enapatternalign_ch2,
	i_errdetect         => rx_errdetect_ch2_rx_errdetect,
	i_disperr           => rx_disperr_ch2_rx_disperr,
	i_reset_n           => resets_n(RESET_BIT_WORDALIGN),--not reset,
	i_clk               => rx_clkout_ch2_clk
);

word_align_ch3 : entity work.rx_align
port map (
	o_data              => data_ch3,
	o_datak             => datak_ch3,
	o_locked            => ch3_align_status,
	i_data              => rx_parallel_data_ch3_rx_parallel_data,
	i_datak             => rx_datak_ch3_rx_datak,
	i_syncstatus        => rx_syncstatus_ch3_rx_syncstatus,
	i_patterndetect     => rx_patterndetect_ch3_rx_patterndetect,
	o_enapatternalign   => open, --enapatternalign_ch3,
	i_errdetect         => rx_errdetect_ch3_rx_errdetect,
	i_disperr           => rx_disperr_ch3_rx_disperr,
	i_reset_n           => resets_n(RESET_BIT_WORDALIGN),--not reset,
	i_clk               => rx_clkout_ch3_clk
);

rdreg_fifo_dma <= (not rdempty_fifo_0) and (not rdempty_fifo_1) and (not rdempty_fifo_2) and (not rdempty_fifo_3);

datainfifo0 : component ip_receiving_fifo
        port map (
            data  	=> data_ch0,  							-- from rx
            wrreq 	=> not datak_ch0(0), 				-- convert xcvr_rx1_datak = '0000' or '0001' to 1 or 0
            rdreq 	=> rdreg_fifo_dma,	 				-- !rdempty0 and ...
            wrclk   	=> rx_clkout_ch0_clk,  				-- to rx clk
            rdclk   	=> pcie_fastclk_out,  							-- to common clk
            aclr  	=> resets_n(RESET_BIT_DATAFIFO), -- for resetting
            q       	=> dma_data_ch0,     				-- goes to dma
            wrfull  	=> open,  								-- should never be 1
            rdempty 	=> rdempty_fifo_0  					-- goes to if for rdreg_fifo_dma
);

datainfifo1 : component ip_receiving_fifo
        port map (
            data  	=> data_ch1,  							-- from rx
            wrreq 	=> not datak_ch1(0), 				-- convert xcvr_rx1_datak = '0000' or '0001' to 1 or 0
            rdreq 	=> rdreg_fifo_dma,	 				-- !rdempty0 and ...
            wrclk   	=> rx_clkout_ch1_clk,  				-- to rx clk
            rdclk   	=> pcie_fastclk_out,  							-- to common clk
            aclr  	=> resets_n(RESET_BIT_DATAFIFO), -- for resetting
            q       	=> dma_data_ch1,     				-- goes to dma
            wrfull  	=> open,  								-- should never be 1
            rdempty 	=> rdempty_fifo_1  					-- goes to if for rdreg_fifo_dma
);

datainfifo2 : component ip_receiving_fifo
        port map (
            data  	=> data_ch2,  							-- from rx
            wrreq 	=> not datak_ch2(0), 				-- convert xcvr_rx1_datak = '0000' or '0001' to 1 or 0
            rdreq 	=> rdreg_fifo_dma,	 				-- !rdempty0 and ...
            wrclk   	=> rx_clkout_ch2_clk,  				-- to rx clk
            rdclk   	=> pcie_fastclk_out,  							-- to common clk
            aclr  	=> resets_n(RESET_BIT_DATAFIFO), -- for resetting
            q       	=> dma_data_ch2,     				-- goes to dma
            wrfull  	=> open,  								-- should never be 1
            rdempty 	=> rdempty_fifo_2  					-- goes to if for rdreg_fifo_dma
);

datainfifo3 : component ip_receiving_fifo
        port map (
            data  	=> data_ch3,  							-- from rx
            wrreq 	=> not datak_ch3(0), 				-- convert xcvr_rx1_datak = '0000' or '0001' to 1 or 0
            rdreq 	=> rdreg_fifo_dma,	 				-- !rdempty0 and ...
            wrclk   	=> rx_clkout_ch3_clk,  				-- to rx clk
            rdclk   	=> pcie_fastclk_out,  							-- to common clk
            aclr  	=> resets_n(RESET_BIT_DATAFIFO), -- for resetting
            q       	=> dma_data_ch3,     				-- goes to dma
            wrfull  	=> open,  								-- should never be 1
            rdempty 	=> rdempty_fifo_3  					-- goes to if for rdreg_fifo_dma
);
	 
--------- PCIe ---------
	 
counter256 : component counter
  port map (
		clk 				=> pcie_fastclk_out,
		reset_n			=> resets_n(RESET_BIT_DATAGEN),
		enable			=>	'1',
		data_en			=>	readmem_wren,
		time_counter 	=> counter_256
);

    e_reset_logic : entity work.reset_logic
    port map (
		rst_n                   => push_button0_db,

		reset_register          => writeregs(RESET_REGISTER_W),
		reset_reg_written       => regwritten(RESET_REGISTER_W),

		resets                  => resets,
		resets_n                => resets_n,

        clk                     => clk--,--clk_125,
    );

    e_version_reg : entity work.version_reg
    port map (
        data_out  => readregs_slow(VERSION_REGISTER_R)(27 downto 0)
    );

--Sync read regs from slow  (125 MHz) to fast (250 MHz) clock
    process(pcie_fastclk_out)
    begin
    if rising_edge(pcie_fastclk_out) then
		clk_sync <= clk;
		clk_last <= clk_sync;
		
		if(clk_sync = '1' and clk_last = '0') then
			readregs(PLL_REGISTER_R) 			<= readregs_slow(PLL_REGISTER_R);
			readregs(VERSION_REGISTER_R) 		<= readregs_slow(VERSION_REGISTER_R);
--			readregs(STATUS_R)(CH0_ALIGN) <= ch0_align_status;
--			readregs(STATUS_R)(CH1_ALIGN) <= ch1_align_status;
--			readregs(STATUS_R)(CH2_ALIGN) <= ch2_align_status;
--			readregs(STATUS_R)(CH3_ALIGN) <= ch3_align_status;
		end if;
		
		readregs(EVENTCOUNTER_REGISTER_R)		<= event_counter;
		readregs(EVENTCOUNTER64_REGISTER_R)		<= event_counter64;
		readregs(TIMECOUNTER_LOW_REGISTER_R)	<= time_counter(31 downto 0);
		readregs(TIMECOUNTER_HIGH_REGISTER_R)	<= time_counter(63 downto 32);
		readregs(EVENT2COUNTER64_REGISTER_R)	<= event2_counter64;
		
--		clk156ch0_sync <= fiforclk;
--		clk156ch0_last <= clk156ch0_sync;
----		clk156ch1_sync <= rx_clkout_ch1_clk;
----		clk156ch1_last <= clk156ch1_sync;
----		clk156ch2_sync <= rx_clkout_ch2_clk;
----		clk156ch2_last <= clk156ch2_sync;
----		clk156ch3_sync <= rx_clkout_ch3_clk;
----		clk156ch3_last <= clk156ch3_sync;
--		if(clk156ch0_sync = '1' and clk156ch0_last = '0') then
--			readregs(STATUS_R)(CH0_ALIGN) <= ch0_align_status;
--			readregs(STATUS_R)(CH1_ALIGN) <= ch1_align_status;
--			readregs(STATUS_R)(CH2_ALIGN) <= ch2_align_status;
--			readregs(STATUS_R)(CH3_ALIGN) <= ch3_align_status;
--		end if;
--		if(clk156ch1_sync = '1' and clk156ch1_last = '0') then
--			readregs(STATUS_R)(CH1_ALIGN) <= ch1_align_status;
--		end if;
--		if(clk156ch2_sync = '1' and clk156ch2_last = '0') then
--			readregs(STATUS_R)(CH2_ALIGN) <= ch2_align_status;
--		end if;
--		if(clk156ch3_sync = '1' and clk156ch3_last = '0') then
--			readregs(STATUS_R)(CH3_ALIGN) <= ch3_align_status;
--		end if;
		
    end if;
    end process;

-- Increase address
process(pcie_fastclk_out, resets_n(RESET_BIT_DATAGEN))
begin
	if(resets_n(RESET_BIT_DATAGEN) = '0') then
		readmem_writeaddr  <= (others => '0');
	elsif(pcie_fastclk_out'event and pcie_fastclk_out = '1') then
		if(readmem_wren = '1') then
			readmem_writeaddr    <= readmem_writeaddr + '1';
			readregs(MEM_WRITEADDR_LOW_REGISTER_R) <= readmem_writeaddr(31 downto 0);
			readregs(MEM_WRITEADDR_HIGH_REGISTER_R) <= readmem_writeaddr(63 downto 32);
		end if;
	end if;
end process;

--Prolong regwritten signals for 50 MHz clock
    process(pcie_fastclk_out)
    begin
    if rising_edge(pcie_fastclk_out) then
		regwritten_del1 <= regwritten_fast;
		regwritten_del2 <= regwritten_del1;
		regwritten_del3 <= regwritten_del2;
		regwritten_del4 <= regwritten_del3;
		for I in 63 downto 0 loop
			if(regwritten_fast(I) = '1' or
				regwritten_del1(I) = '1' or
				regwritten_del2(I) = '1' or
				regwritten_del3(I) = '1' or
				regwritten_del4(I) = '1')
				then
				regwritten(I) 	<= '1';
			else
			regwritten(I) 		<= '0';
			end if;
		end loop;
    end if;
    end process;

  --  LED         :   out std_logic_vector(3 downto 0);
readmem_writeaddr_lowbits <= readmem_writeaddr(15 downto 0);

    e_pcie_block : entity work.pcie_block
    generic map (
		DMAMEMWRITEADDRSIZE => 11,
		DMAMEMREADADDRSIZE  => 11,
		DMAMEMWRITEWIDTH	  => 256
    )
    port map (
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
		writememclk		      => CLK_50_B2J,--input_clk,
		writememreadaddr     => writememreadaddr,
		writememreaddata     => writememreaddata,

		-- pcie readable memory
		readmem_data 			=> readmem_writedata,
		readmem_addr 			=> readmem_writeaddr_lowbits,
		readmemclk				=> pcie_fastclk_out,
		readmem_wren			=> readmem_wren,
		readmem_endofevent	=> readmem_endofevent,

		-- dma memory
		dma_data 				=> dma_data_ch0 & X"DECAFBAD" & dma_data_ch1 & X"DECAFBAD" & dma_data_ch2 & X"DECAFBAD" & dma_data_ch3 & X"DECAFBAD",--counter_256,--rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data & rx_parallel_data,
		dmamemclk				=> pcie_fastclk_out,--rx_clkout_ch0_clk,--rx_clkout_ch0_clk,
		dmamem_wren				=> writeregs(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE) and rdreg_fifo_dma,--'1',
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
		pb_in						=> push_button0_db & push_button1_db & push_button2_db,
		inaddr32_r				=> readregs(inaddr32_r),
		inaddr32_w				=> readregs(inaddr32_w)
    );

end architecture;
