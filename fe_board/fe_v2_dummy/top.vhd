----------------------------------------
-- Dummy version of the Frontend Board 
-- Common Firmware only, no detector-specific parts
-- Martin Mueller, September 2020
----------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.daq_constants.all;
use work.cmp.all;

entity top is 
	port (
		fpga_reset:					in std_logic;
		
		LVDS_clk_si1_fpga_A:					in std_logic; -- 125 MHz base clock for LVDS PLLs - right			//	SI5345
		LVDS_clk_si1_fpga_B:					in std_logic; -- 125 MHz base clock for LVDS PLLs - left				//	SI5345
		transceiver_pll_clock : in std_logic_vector(2 downto 2);--_vector(1 downto 0); -- 125 MHz base clock for transceiver PLLs 	// SI5345
		--extra_transceiver_pll_clocks : in std_logic_vector(1 downto 0); -- 125 MHz base clock for transceiver PLLs	// SI5345
		
		lvds_firefly_clk		 : in std_logic; -- 125 MHz base clock
		
		systemclock				 : in std_logic; -- 50 MHz system clock	// SI5345
		systemclock_bottom	 : in std_logic; -- 50 MHz system clock 	// SI5345
		clk_125_top				 : in std_logic; -- 125 MHz clock spare	//	SI5345
		clk_125_bottom			 : in	std_logic; -- 125 Mhz clock spare	//	SI5345
		spare_clk_osc			 : in std_logic; -- Spare clock	// 50 MHz oscillator
		
		-- Block A: Connections for three chips -- layer 0
		clock_A:				out std_logic;
		data_in_A:			in std_logic_vector(9 downto 1);
		fast_reset_A:		out std_logic;
		SIN_A:				out std_logic;
		
		-- Block B: Connections for three chips -- layer 0
		clock_B:				out std_logic;
		data_in_B:			in std_logic_vector(9 downto 1);
		fast_reset_B:		out std_logic;
		SIN_B:				out std_logic;
		
		-- Block C: Connections for three chips -- layer 1
		clock_C:				out std_logic;
		data_in_C:			in std_logic_vector(9 downto 1);
		fast_reset_C:		out std_logic;
		SIN_C:				out std_logic;
		
		-- Block D: Connections for three chips -- layer 1
		clock_D:				out std_logic;		
		data_in_D:			in std_logic_vector(9 downto 1);
		fast_reset_D:		out std_logic;
		SIN_D:				out std_logic;
		
		-- Block E: Connections for three chips -- layer 1
		clock_E:				out std_logic;
		data_in_E:			in std_logic_vector(9 downto 1);
		fast_reset_E:		out std_logic;
		SIN_E:				out std_logic;
		
		-- Extra signals
		clock_aux:			out std_logic;
		spare_out:			out std_logic_vector(3 downto 2);
		
		-- Fireflies
		firefly1_tx_data: out std_logic_vector(3 downto 0); -- transceiver
		firefly2_tx_data: out std_logic_vector(3 downto 0); -- transceiver 
		firefly1_rx_data:	in  std_logic;                  -- transceiver
		firefly2_rx_data:	in  std_logic_vector(2 downto 0);  -- transceiver
		
		firefly1_lvds_rx_in : in std_logic;--_vector(1 downto 0); -- receiver for slow control or something else
		firefly2_lvds_rx_in : in std_logic;--_vector(1 downto 0); -- receiver for slow control or something else
		
		Firefly_ModSel_n:		out std_logic_vector(1 downto 0);		-- Module select: active low, when host wants to communicate (I2C) with module
		Firefly_Rst_n:			out std_logic_vector(1 downto 0);		-- Module reset: active low, complete reset of module. Module indicates reset done by "low" interrupt_n (data_not_ready is negated).
		Firefly_Scl:			inout std_logic;		-- I2C Clock: module asserts low for clock stretch, timing infos: page 47
		Firefly_Sda:			inout std_logic;	-- I2C Data
	  --Firefly_LPM:			out std_logic;		-- Firefly Low Power Mode: Modules power consumption should be below 1.5 W. active high. Overrideable by I2C commands. Override default: high power (page 19 of documentation).
		Firefly_Int_n:			in std_logic_vector(1 downto 0);		-- Firefly Interrupt: when low: operational fault or status critical. after reset: goes high, and data_not_ready is read with '0' (byte 2 bit 0) and flag field is read
		Firefly_ModPrs_n:		in std_logic_vector(1 downto 0);		-- Module present: Pulled to ground if module is present
		
		-- LEDs, test points and buttons
		PushButton:		in std_logic_vector(1 downto 0);
		FPGA_Test:		out std_logic_vector(7 downto 0);
		
		--LCD
		lcd_csn: 				out 	std_logic;             			--//2.5V    //LCD Chip Select
		lcd_d_cn: 				out 	std_logic;   	      			--//2.5V    //LCD Data / Command Select
		lcd_data: 				out 	std_logic_vector(7 downto 0);   --//2.5V    //LCD Data
		lcd_wen: 				out 	std_logic;             			--//2.5V    //LCD Write Enable
		
		-- SI5345(0): 7 Transceiver clocks @ 125 MHz
		-- SI4345(1): Clocks for the Fibres
		-- 1 reference and 2 inputs for synch
		si45_oe_n:				out std_logic_vector(1 downto 0);	 -- active low output enable -> should always be '0'
		si45_intr_n:			in std_logic_vector(1 downto 0);	-- fault monitor: interrupt pin: change in state of status indicators 
		si45_lol_n:				in std_logic_vector(1 downto 0);	-- fault monitor: loss of lock of DSPLL
		-- I2C sel is set to GND on PCB -> SPI interface		
		si45_rst_n:				out std_logic_vector(1 downto 0);	--	reset
		si45_spi_cs_n:			out std_logic_vector(1 downto 0);	-- chip select
		si45_spi_in:			out std_logic_vector(1 downto 0);	-- data in
		si45_spi_out:			in std_logic_vector(1 downto 0);	-- data out
		si45_spi_sclk:			out std_logic_vector(1 downto 0);	-- clock
		-- change frequency by the FSTEPW parameter
		si45_fdec:				out std_logic_vector(1 downto 0);	-- decrease
		si45_finc:				out std_logic_vector(1 downto 0);	-- increase
			
		-- Midas slow control bus
		mscb_fpga_in:			in std_logic;
		mscb_fpga_out:			out std_logic;
		mscb_fpga_oe_n:		out std_logic;
		
		-- Backplane slot signal
		ref_adr:	in std_logic_vector(7 downto 0);
		
		-- MAX10 IF
		max10_spi_sclk				 : out	std_logic;
		max10_spi_mosi				 : out	std_logic;
		max10_spi_miso				 : inout	std_logic;
		max10_spi_D1				 : inout std_logic;
		max10_spi_D2				 : inout std_logic;
		max10_spi_D3				 : inout std_logic;
		max10_spi_csn				 : out	std_logic
		);
end top;

architecture rtl of top is
 
    -- Debouncers
    signal pb_db                : std_logic_vector(1 downto 0);
    signal version_out          : std_logic_vector(31 downto 0);

    signal counter              : std_logic_vector(31 downto 0);
    signal counter2             : std_logic_vector(31 downto 0);
    
    signal o_firefly2_tx_data   : std_logic_vector(3 downto 0);
    signal o_firefly1_tx_data   : std_logic_vector(3 downto 0);
    
    signal spi_si_MISO          : std_logic;
    signal spi_si_MOSI          : std_logic;
    signal spi_si_SCLK          : std_logic;
    signal spi_si_SS_n          : std_logic_vector(1 downto 0);

begin

    firefly1_tx_data            <= o_firefly1_tx_data;
    firefly2_tx_data            <= o_firefly2_tx_data;
    si45_spi_cs_n               <= spi_si_SS_n;

--v_reg: version_reg 
--    PORT MAP(
--        data_out    => version_out(27 downto 0)
--    );

    db1: entity work.debouncer
    port map(
        i_clk       => spare_clk_osc,
        i_reset_n   => '1',
        i_d(0)      => PushButton(0),
        o_q(0)      => pb_db(0)--,
    );

    db2: entity work.debouncer
    port map(
        i_clk       => spare_clk_osc,
        i_reset_n   => '1',
        i_d(0)      => PushButton(1),
        o_q(0)      => pb_db(1)--,
    );

-- Switch off mscb
    --mscb_fpga_in;
    mscb_fpga_out       <= '0';
    mscb_fpga_oe_n      <= '1';

-- Quad SPI IF to MAX	
-- Tristate bidirectionals for a start
    max10_spi_miso  <= 'Z';
    max10_spi_D1    <= 'Z';
    max10_spi_D2    <= 'Z';
    max10_spi_D3    <= 'Z';

-- Chip select is high	
    max10_spi_csn   <= '1';

-- Lets see if we have a clock
process(transceiver_pll_clock)
begin
if(transceiver_pll_clock(2)'event and transceiver_pll_clock(2) = '1') then
    counter     <= counter +'1';
    lcd_data(0) <= counter(28);
    --lcd_data(1) <= counter(24);
    lcd_data(2) <= counter(29);
    lcd_data(3) <= counter(30);
end if;
end process;

process(spare_clk_osc)
begin
if(spare_clk_osc'event and spare_clk_osc = '1') then
    counter2     <= counter2 +'1';
    --lcd_data(0) <= counter(28);
    --lcd_data(1) <= counter2(27);
    --lcd_data(2) <= counter(29);
    --lcd_data(3) <= counter(30);
end if;
end process;

lcd_data(4) 			<= max10_spi_miso;
lcd_data(5)				<= max10_spi_D1;

lcd_data(6)				<= Firefly_ModPrs_n(1);
lcd_data(7)				<= Firefly_ModPrs_n(0);

    firefly: entity work.firefly
    generic map(
        STARTADDR_g                     => 1--,
    )
    port map(
        i_clk                           => transceiver_pll_clock(2),
        i_sysclk                        => systemclock,
        i_clk_i2c                       => spare_clk_osc,
        o_clk_reco                      => open,
        i_clk_lvds                      => lvds_firefly_clk,
        i_reset_n                       => pb_db(0),
        i_lvds_align_reset_n            => pb_db(1),
        
        --rx
        i_data_fast_serial              => firefly2_rx_data & firefly1_rx_data,
        o_data_fast_parallel            => open,
        o_datak                         => open,
        
        --tx
        o_data_fast_serial(3 downto 0)  => o_firefly1_tx_data,
        o_data_fast_serial(7 downto 4)  => o_firefly2_tx_data,
        i_data_fast_parallel            => x"000000BC"&x"000000BC"&x"000000BC"&x"000000BC" &x"000000BC"&x"000000BC"&x"000000BC"&x"000000BC",
        i_datak                         => "0001000100010001"&"0001000100010001",
        
        --lvds rx
        i_data_lvds_serial              => firefly2_lvds_rx_in & firefly1_lvds_rx_in,
        o_data_lvds_parallel            => open,
        
        --I2C
        i_i2c_enable                    => '1',
        o_Mod_Sel_n                     => Firefly_ModSel_n,
        o_Rst_n                         => Firefly_Rst_n,
        io_scl                          => Firefly_Scl,
        io_sda                          => Firefly_Sda,
        i_int_n                         => Firefly_Int_n,
        i_modPrs_n                      => Firefly_ModPrs_n,
        
        o_testclkout                    => FPGA_Test(5),
        o_testout                       => lcd_data(1)--,
    );

   e_nios: nios
   port map(
      clk_clk                 => spare_clk_osc,
      clk_125_clock_clk       => spare_clk_osc,
      clk_125_reset_reset_n   => pb_db(0),
      clk_156_clock_clk       => spare_clk_osc,
      clk_156_reset_reset_n   => pb_db(0),
      --i2c_sda_in            : in  std_logic                     := 'X';             -- sda_in
      --i2c_scl_in            : in  std_logic                     := 'X';             -- scl_in
      --i2c_sda_oe            : out std_logic;                                        -- sda_oe
      --i2c_scl_oe            : out std_logic;                                        -- scl_oe
      --irq_bridge_irq        : in  std_logic_vector(3 downto 0)  := (others => 'X'); -- irq
      --pio_export            : out std_logic_vector(31 downto 0);                    -- export
      rst_reset_n             => pb_db(0),
      --spi_MISO              : in  std_logic                     := 'X';             -- MISO
      --spi_MOSI              : out std_logic;                                        -- MOSI
      --spi_SCLK              : out std_logic;                                        -- SCLK
      --spi_SS_n              : out std_logic_vector(15 downto 0);                    -- SS_n
      spi_si_MISO             => spi_si_MISO,
      spi_si_MOSI             => spi_si_MOSI,
      spi_si_SCLK             => spi_si_SCLK,
      spi_si_SS_n             => spi_si_SS_n--,
   );

spi_si_MISO <= si45_spi_out(1) when spi_si_SS_n(1)='0' else
               si45_spi_out(0) when spi_si_SS_n(0)='0' else '0';

si45_spi_in(1) <= spi_si_MOSI when spi_si_SS_n(1)='0' else '0';
si45_spi_in(0) <= spi_si_MOSI when spi_si_SS_n(0)='0' else '0';

si45_spi_sclk(1) <= spi_si_SCLK when spi_si_SS_n(1)='0' else '0';
si45_spi_sclk(0) <= spi_si_SCLK when spi_si_SS_n(0)='0' else '0';

si45_rst_n <= (others => '1');
si45_oe_n <= (others => '0');
si45_fdec <= (others => '0');
si45_finc <= (others => '0');
--lcd_data(5 downto 4) <= si45_lol_n;

--FPGA_Test(0) <= Firefly_ModSel_n(0);
--FPGA_Test(1) <= Firefly_Rst_n(0);
--FPGA_Test(2) <= Firefly_Scl;
--FPGA_Test(3) <= Firefly_Sda;
--FPGA_Test(4) <= Firefly_Int_n(0);
--FPGA_Test(5) <= Firefly_ModPrs_n(0);
FPGA_Test(6) <= transceiver_pll_clock(2);
FPGA_Test(7) <= lvds_firefly_clk;

end rtl;
