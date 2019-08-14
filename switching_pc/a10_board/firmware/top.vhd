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
    CLK_100_B3D :   in  std_logic;

    HEX0_D      :   out std_logic_vector(6 downto 0);
    --HEX0_DP     :   out std_logic;

    HEX1_D      :   out std_logic_vector(6 downto 0);
    --HEX1_DP     :   out std_logic;

    LED         :   out std_logic_vector(3 downto 0) := "0000";
    LED_BRACKET :   out std_logic_vector(3 downto 0) := "0000";

    SMA_CLKOUT : out std_logic;
    SMA_CLKIN : in std_logic;

    RS422_DE : out std_logic;
    RS422_DIN : in std_logic; -- 1.8-V
    RS422_DOUT : out std_logic;
    --RS422_RE_n : out std_logic;
    --RJ45_LED_L : out std_logic;
    RJ45_LED_R : out std_logic;
	 
	 --refclk2_qr1_p	: in std_logic;--					1.5-V PCML, default 125MHz
	 --refclk1_qr0_p : in std_logic;-- 1.5-V PCML, default 156.25MHz
	 
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
	 --QSFPB_TX_p          :   out std_logic_vector(3 downto 0);
    
	 QSFPA_RX_p          :   in std_logic_vector(3 downto 0);
	 --QSFPB_RX_p          :   in std_logic_vector(3 downto 0);
    
	 QSFPA_REFCLK_p 		: 	 in std_logic;
	 --QSFPB_REFCLK_p 		: 	 in std_logic;
	 QSFPA_LP_MODE       : out   std_logic;
	 QSFPA_MOD_SEL_n     : out   std_logic;
	 QSFPA_RST_n         : out   std_logic;
    
    
	
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
		 signal input_clk : std_logic;
		 
		 signal reset : std_logic;
		 signal reset_n : std_logic;
		 signal resets : std_logic_vector(31 downto 0);
		 signal resets_n: std_logic_vector(31 downto 0);
		 
		 signal clk_50_cnt : std_logic_vector(31 downto 0);
		 signal clk_125_cnt : std_logic_vector(31 downto 0);

		------------------ Signal declaration ------------------------

		-- pcie
		signal writeregs				: reg32array;
		signal regwritten				: std_logic_vector(63 downto 0);
		signal regwritten_fast		: std_logic_vector(63 downto 0);
		signal regwritten_del1		: std_logic_vector(63 downto 0);
		signal regwritten_del2		: std_logic_vector(63 downto 0);
		signal regwritten_del3		: std_logic_vector(63 downto 0);
		signal regwritten_del4		: std_logic_vector(63 downto 0);
		signal pb_in : std_logic_vector(2 downto 0);
		
		signal readregs				: reg32array;
		signal readregs_slow			: reg32array;

		--//pcie readable memory signals
		signal readmem_writedata 	: std_logic_vector(31 downto 0);
		signal readmem_writeaddr 	: std_logic_vector(63 downto 0);
		signal readmem_writeaddr_finished: std_logic_vector(15 downto 0);
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
		signal dmamemhalffull_counter : std_logic_vector(31 downto 0);
		signal dmamemnothalffull_counter : std_logic_vector(31 downto 0);
		signal endofevent_counter : std_logic_vector(31 downto 0);
		signal notendofevent_counter : std_logic_vector(31 downto 0);

		--//pcie dma memory signals
		signal dma2mem_writedata 	: std_logic_vector(255 downto 0);
		signal dma2mem_wren	 		: std_logic;
		signal dma2mem_endofevent 	: std_logic;
		signal dma2memhalffull 		: std_logic;
		
		-- //pcie fast clock
		signal pcie_fastclk_out		: std_logic;	

		-- //pcie debug signals
		signal pcie_testout				: std_logic_vector(127 downto 0);

		-- Clocksync stuff
		signal clk_sync : std_logic;
		signal clk_last : std_logic;

		-- tranciever ip signals		
		signal tx_clk : std_logic_vector(3 downto 0);
		signal rx_clk : std_logic_vector(3 downto 0);
				
		-- debouncer
		signal push_button0_db : std_logic;
		signal push_button1_db : std_logic;
		signal push_button2_db : std_logic;
		signal push_button3_db : std_logic;
		
		-- data generartor stuff
		signal event_counter : std_logic_vector(31 downto 0);
		signal time_counter : std_logic_vector(63 downto 0);

		-- data generartor64 stuff
		signal event_counter64 : std_logic_vector(31 downto 0);
		signal time_counter64 : std_logic_vector(63 downto 0);
		
		-- sorting
--		signal clk_fast 	: std_logic; -- 312 MHZ
--		signal clks_read : std_logic_vector(4 - 1 downto 0);
--		signal clks_write : std_logic_vector(4 - 1 downto 0);
--		signal fpga_id_in : std_logic_vector(4 * 16 - 1 downto 0);
--		signal enables_in : std_logic_vector(4 - 1 downto 0);
--		signal data_algin : std_logic_vector(63 downto 0);
			
		-- NIOS
		signal flash_ce_n_i : std_logic;
		signal cpu_reset_n_q : std_logic;
		signal i2c_scl_in   : std_logic;
		signal i2c_scl_oe   : std_logic;
		signal i2c_sda_in   : std_logic;
		signal i2c_sda_oe   : std_logic;
		signal flash_tcm_address_out : std_logic_vector(27 downto 0);
		signal wd_rst_n     : std_logic;
		signal cpu_pio_i : std_logic_vector(31 downto 0);
		signal flash_rst_n : std_logic;
		signal debug_nios : std_logic_vector(31 downto 0);
		signal av_qsfp : work.util.avalon_t;
		
		-- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
		signal ZERO : std_logic := '0';
		attribute keep : boolean;
		attribute keep of ZERO : signal is true;
		
		-- data processing
		type fifo_out_array_type is array (3 downto 0) of std_logic_vector(35 downto 0);
		type data_array_type is array (3 downto 0) of std_logic_vector(31 downto 0);
		type datak_array_type is array (3 downto 0) of std_logic_vector(3 downto 0);
		
		signal rx_data : data_array_type;
		signal tx_data : data_array_type;
		signal rx_datak : datak_array_type;
		signal tx_datak : datak_array_type;
		signal rx_data_v : std_logic_vector(4*32-1 downto 0);
		signal rx_datak_v : std_logic_vector(4*4-1 downto 0);
		
		signal idle_ch : std_logic_vector(3 downto 0);

		signal sc_data : data_array_type;
		signal sc_datak : datak_array_type;
		signal sc_ready : std_logic_vector(3 downto 0);
		signal fifo_data : data_array_type;
		signal fifo_datak : datak_array_type;
		signal fifo_wren : std_logic_vector(3 downto 0);
		signal fifo_out : fifo_out_array_type;
		
		signal fifo_read : std_logic;
		signal fifo_empty : std_logic_vector(3 downto 0);
		
		-- Slow Control
		signal mem_data_out : std_logic_vector(127 downto 0);
		signal mem_datak_out : std_logic_vector(15 downto 0);
		
		-- event counter
		signal state_out_eventcounter : std_logic_vector(3 downto 0);
		signal state_out_datagen : std_logic_vector(3 downto 0);
		signal data_pix_generated : std_logic_vector(31 downto 0);
		signal datak_pix_generated : std_logic_vector(3 downto 0);
		signal data_pix_ready : std_logic;
		signal event_length : std_logic_vector(11 downto 0);
		signal dma_data_wren : std_logic;
		signal dma_data : std_logic_vector(255 downto 0);
		signal dma_event_data : std_logic_vector(31 downto 0);
		signal data_counter : std_logic_vector(31 downto 0);
		signal datak_counter : std_logic_vector(3 downto 0);
		
begin 

--------- I/O, CLK, RESET, PLL ---------

clk 		<= CLK_50_B2J;
reset 	<= not push_button0_db;
reset_n 	<= not reset;

pll_125 : component ip_pll_125
  port map (
		outclk_0 	=> SMA_CLKOUT,
		refclk   	=> clk,
		rst      	=> not CPU_RESET_n
);

clk_input : ip_clk_ctrl
  port map (
		inclk  => SMA_CLKIN,
		outclk => input_clk--,
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
	  arst_n => CPU_RESET_n,
	  clk 	=> clk--,
 );

clk_50_cnt_p : process(clk)
begin
	if rising_edge(clk) then
		clk_50_cnt <= clk_50_cnt + 1;
	end if;
end process clk_50_cnt_p;

clk_125_cnt_p : process(input_clk)
begin
	if rising_edge(input_clk) then
		clk_125_cnt <= clk_125_cnt + 1;
	end if;
end process clk_125_cnt_p;

e_segment0 : entity work.hex2seg7
port map (
	i_hex => std_logic_vector(clk_50_cnt)(27 downto 24),
	o_seg => HEX0_D
);

e_segment1 : entity work.hex2seg7
port map (
	i_hex => std_logic_vector(clk_125_cnt)(27 downto 24),
	o_seg => HEX1_D
);

------------- NIOS -------------

nios2 : work.cmp.nios
port map (
	clk_clk                    			=> input_clk,
	
	rst_reset_n                			=> cpu_reset_n_q,

   avm_qsfp_address       					=> av_qsfp.address(13 downto 0),
	avm_qsfp_read          					=> av_qsfp.read,
	avm_qsfp_readdata      					=> av_qsfp.readdata,
	avm_qsfp_write         					=> av_qsfp.write,
	avm_qsfp_writedata     					=> av_qsfp.writedata,
	avm_qsfp_waitrequest   					=> av_qsfp.waitrequest,

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
	
	spi_MISO                   			=> RS422_DIN,
	spi_MOSI                   			=> RS422_DOUT,
	spi_SCLK                   			=> RJ45_LED_R,
	spi_SS_n                   			=> RS422_DE--,
);

QSFPA_LP_MODE <= '0';
QSFPA_MOD_SEL_n <= '1';
QSFPA_RST_n <= '1';

-- generate reset sequence for flash and cpu
reset_ctrl_i : entity work.reset_ctrl
generic map (
		W => 2,
		N => 125 * 10**5 -- 100ms
	)
port map (
		rstout_n(1) => flash_rst_n,
		rstout_n(0) => cpu_reset_n_q,
		rst_n 		=> CPU_RESET_n,-- and wd_rst_n,
		clk 			=> input_clk--,
);

watchdog_i : entity work.watchdog
generic map (
		W => 4,
		N => 125 * 10**6 -- 1s
)
port map (
		d 			=> cpu_pio_i(3 downto 0),

		rstout_n => wd_rst_n,

		rst_n 	=> CPU_RESET_n,
		clk 		=> input_clk--,
);

LED(0) <= cpu_pio_i(7);
LED(1) <= not cpu_reset_n_q;
LED(2) <= not flash_rst_n;
LED(3) <= '0';

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

e_qsfp : entity work.xcvr_a10
port map (
    i_tx_data   => X"03CAFEBC"
                 & X"02CAFEBC"
                 & tx_data(1)
                 & tx_data(0),
    i_tx_datak  => "0001"
                 & "0001"
                 & tx_datak(1)
                 & tx_datak(0),

    o_rx_data   => rx_data_v,
    o_rx_datak  => rx_datak_v,

    o_tx_clkout => tx_clk,
    i_tx_clkin  => (others => tx_clk(0)),
    o_rx_clkout => open,--rx_clk,
    i_rx_clkin  => (others => tx_clk(0)),

    o_tx_serial => QSFPA_TX_p,
    i_rx_serial => QSFPA_RX_p,

    i_pll_clk   => input_clk,
    i_cdr_clk   => input_clk,

    i_avs_address     => av_qsfp.address(13 downto 0),
    i_avs_read        => av_qsfp.read,
    o_avs_readdata    => av_qsfp.readdata,
    i_avs_write       => av_qsfp.write,
    i_avs_writedata   => av_qsfp.writedata,
    o_avs_waitrequest => av_qsfp.waitrequest,

    i_reset     => not CPU_RESET_n,
    i_clk       => input_clk--,
);
--assign vector types to array types for qsfp rx signals
rx_data(3)<=rx_data_v(32*4-1 downto 32*3);
rx_data(2)<=rx_data_v(32*3-1 downto 32*2);
rx_data(1)<=rx_data_v(32*2-1 downto 32*1);
rx_data(0)<=rx_data_v(32*1-1 downto 32*0);
rx_datak(3)<=rx_datak_v(4*4-1 downto 4*3);
rx_datak(2)<=rx_datak_v(4*3-1 downto 4*2);
rx_datak(1)<=rx_datak_v(4*2-1 downto 4*1);
rx_datak(0)<=rx_datak_v(4*1-1 downto 4*0);

------------- data demerger and fifos -------------

--fifo_read <= (not fifo_empty(0)) and (not fifo_empty(1)) and (not fifo_empty(2)) and (not fifo_empty(3));

--process(tx_clk(0), reset_n)
--begin
--	if(reset_n = '0') then
--		idle_ch <= (others => '0');
--	elsif(rising_edge(tx_clk(0))) then
--		idle_ch <= (others => '0');
--		if(rx_data(0) = x"000000BC" and rx_datak(0) = "0001") then
--			idle_ch(0) <= '1';
--		end if;
--		if(rx_data(1) = x"000000BC" and rx_datak(1) = "0001") then
--			idle_ch(1) <= '1';
--		end if;
--		if(rx_data(2) = x"000000BC" and rx_datak(2) = "0001") then
--			idle_ch(2) <= '1';
--		end if;
--		if(rx_data(3) = x"000000BC" and rx_datak(3) = "0001") then
--			idle_ch(3) <= '1';
--		end if;
--	end if;
--end process;

--fifo_demerge :
--for i in 0 to 3 generate
----		data_demerger : data_demerge
----			port map(
----				clk				=> tx_clk(0),			-- receive clock (156.25 MHz)
----				reset				=> not reset_n,
----				aligned			=> '1',					-- word alignment achieved
----				data_in			=>	rx_data(i),			-- optical from frontend board
----				datak_in			=> rx_datak(i),
----				data_out			=> fifo_data(i),		-- to sorting fifos
----				data_ready		=>	fifo_wren(i),	  	-- write req for sorting fifos
----				datak_out      => fifo_datak(i),
----				sc_out			=> sc_data(i),			-- slowcontrol from frontend board
----				sc_out_ready	=> sc_ready(i),
----				fpga_id			=> open,					-- FPGA ID of the connected frontend board
----				sck_out      	=> sc_datak(i)--,
----		);
--		
--	fifo : transceiver_fifo
--		port map (
--			data    => rx_data(i) & rx_datak(i), --fifo_data_in_ch0 & fifo_datak_in_ch0,
--			wrreq   => not idle_ch(i),
--			rdreq   => not fifo_empty(i),
--			wrclk   => tx_clk(0),--rx_clk(i),
--			rdclk   => pcie_fastclk_out,
--			aclr    => not reset_n,
--			q       => fifo_out(i),
--			rdempty => fifo_empty(i),
--			wrfull  => open--,
--	);
--end generate fifo_demerge;


------------- Event Counter ------------------

e_data_gen : entity work.data_generator_a10
	port map (
		clk 						=> tx_clk(0),
		reset						=> resets(RESET_BIT_DATAGEN),
		enable_pix	         => writeregs(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE_PIXEL),
		random_seed 			=> (others => '1'),
		data_pix_generated   => data_pix_generated,
		datak_pix_generated  => datak_pix_generated,
		data_pix_ready			=>	data_pix_ready,
		start_global_time		=> (others => '0'),
		slow_down				=> writeregs(DMA_SLOW_DOWN_REGISTER_W),
		state_out				=> state_out_datagen--,
);

process(tx_clk(0), reset_n)
begin
	if(reset_n = '0') then
		data_counter 	<= (others => '0');
		datak_counter 	<= (others => '0');
	else
		if (writeregs(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE_PIXEL) = '1') then
			data_counter 	<= data_pix_generated;
			datak_counter 	<= datak_pix_generated;
		else
			data_counter 	<= rx_data(0);
			datak_counter 	<= rx_datak(0);
		end if;
	end if;
end process;

e_event_counter : entity work.event_counter
	port map(
		clk						=> tx_clk(0),
		dma_clk					=> pcie_fastclk_out,
		reset_n					=> resets_n(RESET_BIT_EVENT_COUNTER),
		rx_data					=> data_counter,
		rx_datak					=> datak_counter,
		dma_wen_reg				=> writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE),
		event_length			=> event_length,
		dma_data_wren			=> dma_data_wren,
		dmamem_endofevent		=> dmamem_endofevent,
		dma_data					=> dma_event_data,
		state_out				=> state_out_eventcounter--,
);
	
dma_data <=	X"00000" & event_length &
				dma_event_data 			&
				x"1ABACAF" & state_out_eventcounter &
				x"2ABACAF" & state_out_datagen &
				x"3ABACAFE" &
				x"4ABACAFE" &
				x"5ABACAFE" &
				x"6ABACAFE";

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

------------- Slow Control -------------

master : sc_master
	generic map(
		NLINKS => 4
	)
	port map(
		clk					=> tx_clk(0),
		reset_n				=> resets_n(RESET_BIT_SC_MASTER),
		enable				=> '1',
		mem_data_in			=> writememreaddata,
		mem_addr				=> writememreadaddr,
		mem_data_out		=> mem_data_out,
		mem_data_out_k		=> mem_datak_out,
		done					=> open,
		stateout				=> open--,
);

slave : sc_slave
	port map(
		clk							=> tx_clk(0),
		reset_n						=> resets_n(RESET_BIT_SC_SLAVE),
		enable						=> '1',
		link_data_in				=> rx_data(0),
		link_data_in_k				=> rx_datak(0),
		mem_addr_out				=> readmem_writeaddr(15 downto 0),
		mem_addr_finished_out   => readmem_writeaddr_finished,
		mem_data_out				=> readmem_writedata,
		mem_wren						=> readmem_wren,
		stateout						=> LED_BRACKET--,
);

tx_data(0) <= mem_data_out(31 downto 0);
tx_datak(0) <= mem_datak_out(3 downto 0);

------------- PCIe -------------

resetlogic : entity work.reset_logic
	port map(
		clk                     => tx_clk(0),--clk,
		rst_n                   => push_button0_db,

		reset_register          => writeregs(RESET_REGISTER_W),
		reset_reg_written       => regwritten(RESET_REGISTER_W),

		resets                  => resets,
		resets_n                => resets_n--,                                                           
);

vreg : entity work.version_reg
	port map(
		data_out  => readregs_slow(VERSION_REGISTER_R)(27 downto 0)
);

--Sync read regs from slow  (50 MHz) to fast (250 MHz) clock
process(pcie_fastclk_out)
begin
	if(pcie_fastclk_out'event and pcie_fastclk_out = '1') then
		clk_sync <= tx_clk(0);--clk;
		clk_last <= clk_sync;
		
		if(clk_sync = '1' and clk_last = '0') then
			readregs(PLL_REGISTER_R) 					<= readregs_slow(PLL_REGISTER_R);
			readregs(VERSION_REGISTER_R) 				<= readregs_slow(VERSION_REGISTER_R);
		end if;
		
		readregs(EVENTCOUNTER_REGISTER_R)			<= event_counter;
		readregs(EVENTCOUNTER64_REGISTER_R)			<= event_counter64;
		
		readregs(DMA_STATUS_R)(DMA_DATA_WEN)		<= dma_data_wren;
		
		readregs(DMA_HALFFUL_REGISTER_R)				<= dmamemhalffull_counter;
		readregs(DMA_NOTHALFFUL_REGISTER_R)			<= dmamemnothalffull_counter;
		
		readregs(DMA_ENDEVENT_REGISTER_R)			<= endofevent_counter;
		readregs(DMA_NOTENDEVENT_REGISTER_R)		<= notendofevent_counter;
		
		readregs(TIMECOUNTER_LOW_REGISTER_R)		<= time_counter(31 downto 0);
		readregs(TIMECOUNTER_HIGH_REGISTER_R)		<= time_counter(63 downto 32);

		readregs(MEM_WRITEADDR_HIGH_REGISTER_R) <= (others => '0');
		readregs(MEM_WRITEADDR_LOW_REGISTER_R) <= (X"0000" & readmem_writeaddr_finished);
	end if;
end process;

e_dma_evaluation : entity work.dma_evaluation
   port map(
		clk							=> pcie_fastclk_out,
		reset_n						=> resets_n(RESET_BIT_DMA_EVAL),
		dmamemhalffull				=> dmamemhalffull,
		dmamem_endofevent			=> dmamem_endofevent,
		halffull_counter			=> dmamemhalffull_counter,
		nothalffull_counter		=> dmamemnothalffull_counter,
		endofevent_counter		=> endofevent_counter,
		notendofevent_counter	=> notendofevent_counter--,
	);

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

readmem_writeaddr_lowbits 	<= readmem_writeaddr(15 downto 0);
pb_in 							<= push_button0_db & push_button1_db & push_button2_db;

pcie_b : entity work.pcie_block 
	generic map(
		DMAMEMWRITEADDRSIZE 	=> 11,
		DMAMEMREADADDRSIZE  	=> 11,
		DMAMEMWRITEWIDTH	  	=> 256
	)
	port map(
		local_rstn				=> '1',--resets_n(RESET_BIT_PCIE_LOCAL),
		appl_rstn				=> '1',--resets_n(RESET_BIT_PCIE),
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
		comp_led			    	=> open,
		L0_led			      => open,

		-- pcie registers (write / read register, readonly, read write, in tools/dmatest/rw) -Sync read regs 
		writeregs		      => writeregs,
		regwritten		      => regwritten_fast,
		readregs			    	=> readregs,

		-- pcie writeable memory
		writememclk		      => tx_clk(0),
		writememreadaddr     => writememreadaddr,
		writememreaddata     => writememreaddata,

		-- pcie readable memory
		readmem_data 			=> readmem_writedata,
		readmem_addr 			=> readmem_writeaddr_lowbits,
		readmemclk				=> tx_clk(0),
		readmem_wren			=> readmem_wren,
		readmem_endofevent	=> readmem_endofevent,

		-- dma memory 
		dma_data 				=> dma_data,
		dmamemclk				=> pcie_fastclk_out,
		dmamem_wren				=> dma_data_wren,
		dmamem_endofevent		=> dmamem_endofevent,
		dmamemhalffull			=> dmamemhalffull,

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
		inaddr32_w				=> readregs(inaddr32_w)--,
);

end;
