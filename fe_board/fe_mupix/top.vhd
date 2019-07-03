library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.reg_map_s4.all;

entity top is
port (
    
    -- SI45

    si45_oe_n       : out   std_logic; -- <= '0'
    si45_rst_n      : out   std_logic; -- reset
    si45_spi_out    : in    std_logic; -- slave data out
    si45_spi_in     : out   std_logic; -- slave data in
    si45_spi_sclk   : out   std_logic; -- clock
    si45_spi_cs_n   : out   std_logic; -- chip select


    -- QSFP

    qsfp_pll_clk    : in    std_logic; -- 156.25 MHz

    QSFP_ModSel_n   : out   std_logic; -- module select (i2c)
    QSFP_Rst_n      : out   std_logic;
    QSFP_LPM        : out   std_logic; -- Low Power Mode

    qsfp_tx         : out   std_logic_vector(3 downto 0);
    qsfp_rx         : in    std_logic_vector(3 downto 0);



    -- POD

--    pod_pll_clk     : in    std_logic;
--
--    pod_tx_reset    : out   std_logic;
--    pod_rx_reset    : out   std_logic;
--
--    pod_tx          : out   std_logic_vector(3 downto 0);
--    pod_rx          : in    std_logic_vector(3 downto 0);


	 -- Block A here : Connections for two MuPix8 via SCSI adapter card 
	 clock_A					: out std_logic;
	 data_in_A_0			: in std_logic_vector(3 downto 0);
	 data_in_A_1			: in std_logic_vector(3 downto 0);
	 fast_reset_A			: out std_logic;
	 test_pulse_A			: out std_logic;
 
	 CTRL_SDO_A				: in std_logic; -- A_ctrl_dout_front
	 CTRL_SDI_A				: out std_logic; -- A_ctrl_din_front
	 CTRL_SCK1_A			: out std_logic; -- A_ctrl_clk1_front
	 CTRL_SCK2_A			: out std_logic; -- A_ctrl_clk2_front
	 CTRL_RB_A				: out std_logic; -- A_ctrl_rb_front
	 CTRL_Load_A			: out std_logic; -- A_ctrl_ld_front
	 
	 -- A_trig_front
	 chip_reset_A			: out std_logic; -- is called trigger on adapter card!
 
	 SPI_DIN0_A				: out std_logic; -- A_spi_din_front
	 SPI_DIN1_A				: out std_logic; -- A_spi_din_back
	 SPI_CLK_A				: out std_logic; -- A_spi_clk_front
	 SPI_LD_DAC_A			: out std_logic; -- A_spi_ld_front
	 SPI_LD_ADC_A			: out std_logic; -- A_spi_ld_tmp_dac_front
	 SPI_LD_TEMP_DAC_A	: out std_logic; -- A_spi_ld_adc_front
	 SPI_DOUT_ADC_0_A		: in std_logic; -- A_spi_dout_adc_front
	 SPI_DOUT_ADC_1_A		: in std_logic; -- A_spi_dout_adc_back
	 
    --

    led_n       : out   std_logic_vector(15 downto 0);

    reset_n     : in    std_logic;
    -- 125 MHz
    clk_aux     : in    std_logic--;
);
end entity;

architecture arch of top is

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    signal led : std_logic_vector(led_n'range);

    signal nios_clk, nios_reset_n : std_logic;
    signal nios_pio : std_logic_vector(31 downto 0);

    signal i2c_scl_in, i2c_scl_oe, i2c_sda_in, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n : std_logic_vector(1 downto 0);

    signal malibu_clk : std_logic;
    signal malibu_rx_data_clk : std_logic;
    signal malibu_rx_data : std_logic_vector(15 downto 0);
    signal malibu_rx_datak : std_logic_vector(1 downto 0);
    signal malibu_word : std_logic_vector(47 downto 0);

    signal avm_pod, avm_qsfp : work.mu3e.avalon_t;

    signal qsfp_tx_data : std_logic_vector(127 downto 0);
    signal qsfp_tx_datak : std_logic_vector(15 downto 0);

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);

    signal qsfp_reset_n : std_logic;

    signal avm_sc : work.mu3e.avalon_t;

    signal ram_addr_a : std_logic_vector(15 downto 0);
    signal ram_rdata_a : std_logic_vector(31 downto 0);
    signal ram_wdata_a : std_logic_vector(31 downto 0);
    signal ram_we_a : std_logic;

    signal data_to_fifo : std_logic_vector(35 downto 0);
    signal data_to_fifo_we : std_logic;
    signal data_from_fifo : std_logic_vector(35 downto 0);
    signal data_from_fifo_re : std_logic;
    signal data_from_fifo_empty : std_logic;

    signal sc_to_fifo : std_logic_vector(35 downto 0);
    signal sc_to_fifo_we : std_logic;
    signal sc_from_fifo : std_logic_vector(35 downto 0);
    signal sc_from_fifo_re : std_logic;
    signal sc_from_fifo_empty : std_logic;
	 
	 signal writememreaddata_mp8 : std_logic_vector(31 downto 0);
	 signal writememreadaddr_mp8 : std_logic_vector(15 downto 0);
	 signal mp8_busy_n : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_mem_data_out : std_logic_vector(31 downto 0);
	 signal mp8_wren : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ld : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_rb : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ctrl_dout : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ctrl_din : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ctrl_clk1 : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ctrl_clk2 : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ctrl_ld : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_ctrl_rb : std_logic_vector(0 downto 0); -- NCHIPS
	 signal mp8_dataout : std_logic_vector(31 downto 0); -- NCHIPS
	 
	 -- SPI
	 signal A_spi_wren_front :	std_logic_vector(2 downto 0);
	 signal A_spi_busy_n_front : std_logic;
	 signal A_spi_ldn_front : std_logic_vector(2 downto 0);
	 signal A_spi_sdo_front : std_logic_vector(2 downto 0);
	 
	 -- SPI output signals A front
    signal	injection1_out_A_front:			std_logic_vector(15 downto 0);
    signal	threshold_pix_out_A_front:		std_logic_vector(15 downto 0);	
    signal	threshold_low_out_A_front:		std_logic_vector(15 downto 0);
    signal	threshold_high_out_A_front:	std_logic_vector(15 downto 0);
    signal	temp_dac_out_A_front:			std_logic_vector(15 downto 0);
    signal	temp_adc_out_A_front:			std_logic_vector(31 downto 0);
    -- SPI output signals A back
    signal	injection1_out_A_back:			std_logic_vector(15 downto 0);
    signal	threshold_pix_out_A_back:		std_logic_vector(15 downto 0);	
    signal	threshold_low_out_A_back:		std_logic_vector(15 downto 0);
    signal	threshold_high_out_A_back:		std_logic_vector(15 downto 0);
    signal	temp_dac_out_A_back:				std_logic_vector(15 downto 0);
    signal	temp_adc_out_A_back:				std_logic_vector(31 downto 0);
     
	 subtype reg32 is std_logic_vector(31 downto 0);
	 type reg32array is array (36 downto 0) of reg32;
	 signal fpga_reg32 : reg32array;
	 
	 signal add_reg_ram : std_logic_vector(13 downto 0);
	 signal data_reg_ram : std_logic_vector(31 downto 0);
	 signal wen_reg_ram : std_logic;
	 signal wdate_reg_ram : std_logic_vector(31 downto 0);
	 type state_spi is (waiting, starting, read_out_pix, write_pix, read_out_th, ending);
    signal spi_state : state_spi;
	  

begin

    led_n <= not led;

    -- 125 MHz
    i_clk_aux_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( clkout => led(15), rst_n => reset_n, clk => clk_aux );

    -- 156.25 MHz
    i_clk_qsfp_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( clkout => led(14), rst_n => reset_n, clk => qsfp_pll_clk );

    ----------------------------------------------------------------------------
    -- NIOS

    -- 50 MHz
    i_nios_clk : entity work.ip_altpll
    generic map (
        DIV => 5,
        MUL => 2--,
    )
    port map (
        c0 => nios_clk,
        locked => open,
        areset => '0',
        inclk0 => clk_aux--,
    );

    i_nios_reset_n : entity work.reset_sync
    port map ( rstout_n => nios_reset_n, arst_n => reset_n, clk => nios_clk );

    i_qsfp_reset_n : entity work.reset_sync
    port map ( rstout_n => qsfp_reset_n, arst_n => reset_n, clk => qsfp_pll_clk );

    led(12) <= nios_pio(7);

    i_nios : component work.cmp.nios
    port map (
        avm_qsfp_address        => avm_qsfp.address(15 downto 0),
        avm_qsfp_read           => avm_qsfp.read,
        avm_qsfp_readdata       => avm_qsfp.readdata,
        avm_qsfp_write          => avm_qsfp.write,
        avm_qsfp_writedata      => avm_qsfp.writedata,
        avm_qsfp_waitrequest    => avm_qsfp.waitrequest,

        avm_pod_address         => avm_pod.address(15 downto 0),
        avm_pod_read            => avm_pod.read,
        avm_pod_readdata        => avm_pod.readdata,
        avm_pod_write           => avm_pod.write,
        avm_pod_writedata       => avm_pod.writedata,
        avm_pod_waitrequest     => avm_pod.waitrequest,

        avm_sc_address          => avm_sc.address(15 downto 0),
        avm_sc_read             => avm_sc.read,
        avm_sc_readdata         => avm_sc.readdata,
        avm_sc_write            => avm_sc.write,
        avm_sc_writedata        => avm_sc.writedata,
        avm_sc_waitrequest      => avm_sc.waitrequest,

        sc_clk_clk          => qsfp_pll_clk,
        sc_reset_reset_n    => qsfp_reset_n,

        --
        -- nios base
        --

        i2c_scl_in => i2c_scl_in,
        i2c_scl_oe => i2c_scl_oe,
        i2c_sda_in => i2c_sda_in,
        i2c_sda_oe => i2c_sda_oe,

        spi_miso => spi_miso,
        spi_mosi => spi_mosi,
        spi_sclk => spi_sclk,
        spi_ss_n => spi_ss_n,

        pio_export => nios_pio,

        rst_reset_n => nios_reset_n,
        clk_clk => nios_clk--,
    );

    si45_oe_n <= '0';
    si45_rst_n <= '1';
    si45_spi_in <= spi_mosi;
--    spi_miso <= si45_spi_out;
    si45_spi_sclk <= spi_sclk;
    si45_spi_cs_n <= spi_ss_n(0);



    -- I2C
    i2c_scl_in <= not i2c_scl_oe;
    i2c_sda_in <=
        malibu_i2c_sda and
        '1';
    malibu_i2c_scl <= ZERO when i2c_scl_oe = '1' else 'Z';
    malibu_i2c_sda <= ZERO when i2c_sda_oe = '1' else 'Z';

    -- SPI
    malibu_spi_sdi <= spi_mosi;
--    spi_miso <= malibu_spi_sdo;
    malibu_spi_sck <= spi_sclk;

    spi_miso <= si45_spi_out when spi_ss_n(0) = '0' else
                malibu_spi_sdo when spi_ss_n(1) = '0' else '0';

    ----------------------------------------------------------------------------
    -- QSFP

    QSFP_ModSel_n <= '1';
    QSFP_Rst_n <= '1';
    QSFP_LPM <= '0';

    i_qsfp : entity work.xcvr_s4
    generic map (
        data_rate => 6250,
        pll_freq => 156.25--,
    )
    port map (
        -- avalon slave interface
        avs_address     => avm_qsfp.address(15 downto 2),
        avs_read        => avm_qsfp.read,
        avs_readdata    => avm_qsfp.readdata,
        avs_write       => avm_qsfp.write,
        avs_writedata   => avm_qsfp.writedata,
        avs_waitrequest => avm_qsfp.waitrequest,

        tx_data     => qsfp_tx_data,
        tx_datak    => qsfp_tx_datak,

        rx_data     => qsfp_rx_data,
        rx_datak    => qsfp_rx_datak,

        tx_clkout   => open,
        tx_clkin    => (others => qsfp_pll_clk),
        rx_clkout   => open,
        rx_clkin    => (others => qsfp_pll_clk),

        tx_p        => qsfp_tx,
        rx_p        => qsfp_rx,

        pll_refclk  => qsfp_pll_clk,
        cdr_refclk  => qsfp_pll_clk,

        reset   => not nios_reset_n,
        clk     => nios_clk--,
    );

    qsfp_tx_data(127 downto 32) <=
          X"03CAFE" & work.util.D28_5
        & X"02BABE" & work.util.D28_5
        & X"01DEAD" & work.util.D28_5;

    qsfp_tx_datak(15 downto 4) <=
          "0001"
        & "0001"
        & "0001";

    ----------------------------------------------------------------------------



    ----------------------------------------------------------------------------
    -- SLOW CONTROL

    i_sc_ram1 : entity work.ip_ram -- nios
    generic map (
        ADDR_WIDTH => 14,
        DATA_WIDTH => 32--,
    )
    port map (
        address_b   => avm_sc.address(15 downto 2),
        q_b         => open,--avm_sc.readdata,
        wren_b      => avm_sc.write,
        data_b      => avm_sc.writedata,
        clock_b     => qsfp_pll_clk,

        address_a   => ram_addr_a(13 downto 0),
        q_a         => ram_rdata_a,
        wren_a      => ram_we_a,--ram_we,
        data_a      => ram_wdata_a,
        clock_a     => qsfp_pll_clk--,
    );
    avm_sc.waitrequest <= '0';

	 i_sc_ram2 : entity work.ip_ram -- pixel
    generic map (
        ADDR_WIDTH => 14,
        DATA_WIDTH => 32--,
    )
    port map (
        address_b   => avm_sc.address(15 downto 2),
        q_b         => avm_sc.readdata,
        wren_b      => avm_sc.write,
        data_b      => avm_sc.writedata,
        clock_b     => qsfp_pll_clk,

        address_a   => add_reg_ram,
        q_a         => data_reg_ram,
        wren_a      => wen_reg_ram,
        data_a      => wdate_reg_ram,
        clock_a     => qsfp_pll_clk--,
    );
	 
    i_sc : entity work.sc_s4
    port map (
        clk 				=> qsfp_pll_clk,
        reset_n 			=> reset_n,
        enable 			=> '1',

        mem_data_in 		=> ram_rdata_a,

        link_data_in 	=> qsfp_rx_data(31 downto 0),
        link_data_in_k 	=> qsfp_rx_datak(3 downto 0),

        fifo_data_out 	=> sc_to_fifo,
        fifo_we 			=> sc_to_fifo_we,

        mem_data_out 	=> ram_wdata_a,
        mem_addr_out 	=> ram_addr_a,
        mem_wren 			=> ram_we_a,

        stateout 			=> open--,
    );
	 
--	 with add_reg_ram(13 downto 6) select ram_we <= 
--		ram_we_a <= when x"00";
--		
--	 with add_reg_ram(5 downto 0) select reg_we <=
--		ram_we_a < when "000000";
--	
--	 process(qsfp_pll_clk, reset_n)
--	 begin
--		if(reset_n = '0') then
--			fpga_reg32 <=  (others => (others => '0'));
--		elsif(rising_edge(qsfp_pll_clk)) then
--			if(reg_we = '1') then
--				fpga_reg32(to_integer(add_reg_ram(13 downto 0))) <= ram_wdata_a;
--			end if;
		
    ----------------------------------------------------------------------------
    -- MUPIX 8 Slow Control and SPI for DACs and ADC's
	 
	 
	 process(qsfp_pll_clk, reset_n) -- set_reg 
	 begin
		if(reset_n = '0') then
			add_reg_ram <= (others => '0');
			wdate_reg_ram <= (others => '0');
			A_spi_wren_front <= (others => '0');
			wen_reg_ram <= '0';
			spi_state <= waiting;
		elsif(rising_edge(qsfp_pll_clk)) then
			wdate_reg_ram <= (others => '0');
			A_spi_wren_front <= (others => '0');
			wen_reg_ram <= '0';
			
			case spi_state is
				when waiting =>
					if(data_reg_ram = x"00000001") then
						add_reg_ram(3 downto 0) <= x"1";
						spi_state <= starting;
					end if;
					
				when starting =>
					add_reg_ram(3 downto 0) <= x"2";
					fpga_reg32(THRESHOLD_DAC_A_FRONT_REGISTER_W)(THRESHOLD_LOW_RANGE) <= data_reg_ram(15 downto 0);
					fpga_reg32(THRESHOLD_DAC_A_FRONT_REGISTER_W)(THRESHOLD_HIGH_RANGE) <= data_reg_ram(31 downto 16);
					spi_state <= write_pix;		
			
				when write_pix =>
					A_spi_wren_front <= "001";
					fpga_reg32(INJECTION_DAC_A_FRONT_REGISTER_W)(INJECTION1_RANGE) <= data_reg_ram(15 downto 0);
					fpga_reg32(INJECTION_DAC_A_FRONT_REGISTER_W)(THRESHOLD_PIX_RANGE) <= data_reg_ram(31 downto 16); 
					spi_state <= read_out_th;		
		
				when read_out_th =>
					add_reg_ram(3 downto 0) <= x"3";
					wen_reg_ram <= '1';
					wdate_reg_ram(15 downto 0) <= threshold_low_out_A_front;
					wdate_reg_ram(31 downto 16) <= threshold_high_out_A_front;
					spi_state <= read_out_pix;
					
				when read_out_pix =>
					add_reg_ram(3 downto 0) <= x"4";
					wen_reg_ram <= '1';
					wdate_reg_ram(15 downto 0) <= injection1_out_A_front;
					wdate_reg_ram(31 downto 16) <= threshold_pix_out_A_front;
					spi_state <= ending;
					
				when ending =>
					add_reg_ram <= (others => '0');
					wdate_reg_ram <= (others => '0');
					wen_reg_ram <= '1';
					spi_state <= waiting;
					
				when others =>
					spi_state <= waiting;
					add_reg_ram <= (others => '0');
			
			end case;
			
		end if;
	end process;
	 
	 i_sc_mp8_master : work.mp8_sc_master
	 generic map(NCHIPS => 1)
	 port map (
		  clk				=> qsfp_pll_clk,
		  reset_n		=> reset_n,
	 	  mem_data_in	=> writememreaddata_mp8,
		  busy_n			=> mp8_busy_n,
		
		  mem_addr		=> writememreadaddr_mp8,
		  mem_data_out	=> mp8_mem_data_out,
		  wren			=> mp8_wren,
		  ctrl_ld		=> mp8_ld,
		  ctrl_rb		=> mp8_rb,
		  done			=> open, 
		  stateout		=> open--,
	 );
	 
	 gen_slowc:
	 for i in 0 to 1-1 generate -- nchips
	 i_mp8_sc : work.mp8_slowcontrol
	 port map(
		  clk				=> qsfp_pll_clk,
		  reset_n		=> reset_n,
		  ckdiv			=> (others => '0'), -- this need to be set to a register
		  mem_data		=> mp8_mem_data_out,
		  wren			=> mp8_wren(i),
		  ld_in			=> mp8_ld(i),
		  rb_in			=> mp8_rb(i),
		  ctrl_dout		=> mp8_ctrl_dout(i),
		  ctrl_din		=> mp8_ctrl_din(i),
		  ctrl_clk1		=> mp8_ctrl_clk1(i),
		  ctrl_clk2		=> mp8_ctrl_clk2(i),
		  ctrl_ld		=> mp8_ctrl_ld(i),
		  ctrl_rb		=> mp8_ctrl_rb(i),
		  busy_n			=> mp8_busy_n(i),
		  dataout		=> mp8_dataout--, need also be generated via nchips
	 );	
	 end generate gen_slowc;
		 
	 process(qsfp_pll_clk)
	 begin
		if(rising_edge(qsfp_pll_clk))then	
			mp8_ctrl_dout(0)	<= CTRL_SDO_A;
		end if;
	 end process;
	 
	 process(qsfp_pll_clk)
	 begin
		if(rising_edge(qsfp_pll_clk))then	
			CTRL_SDI_A		<= mp8_ctrl_din(0);
			CTRL_SCK1_A		<= mp8_ctrl_clk1(0);
			CTRL_SCK2_A		<= mp8_ctrl_clk2(0);
			CTRL_Load_A		<= mp8_ctrl_ld(0);
			CTRL_RB_A		<= mp8_ctrl_rb(0);
		end if;
	 end process;
	 
	 A_spi_sdo_front 			<= SPI_DOUT_ADC_0_A & "00";-- A_spi_dout_dac_front & A_dac4_dout_front;
	 SPI_LD_ADC_A 				<= A_spi_ldn_front(2);
    SPI_LD_TEMP_DAC_A 		<= A_spi_ldn_front(1);
	 SPI_LD_DAC_A 				<= A_spi_ldn_front(0);
	 --A_spi_wren_front 		<= fpga_reg32(DAC_WRITE_REGISTER_W)(DAC_WRITE_A_FRONT_RANGE);

	 ip_spi_master_mupix : entity work.spi_master 
	 port map(
		clk					=> qsfp_pll_clk,
		reset_n				=> reset_n,
		injection1_reg		=> fpga_reg32(INJECTION_DAC_A_FRONT_REGISTER_W)(INJECTION1_RANGE),
		threshold_pix_reg	=> fpga_reg32(INJECTION_DAC_A_FRONT_REGISTER_W)(THRESHOLD_PIX_RANGE),
		threshold_low_reg	=> fpga_reg32(THRESHOLD_DAC_A_FRONT_REGISTER_W)(THRESHOLD_LOW_RANGE),
		threshold_high_reg=> fpga_reg32(THRESHOLD_DAC_A_FRONT_REGISTER_W)(THRESHOLD_HIGH_RANGE),
		temp_dac_reg		=> fpga_reg32(TEMP_A_FRONT_REGISTER_W)(TEMP_DAC_RANGE),
		temp_adc_reg		=> fpga_reg32(TEMP_A_FRONT_REGISTER_W)(TEMP_ADC_W_RANGE),	
		wren					=> A_spi_wren_front,
		busy_n				=> A_spi_busy_n_front,
		spi_sdi				=> SPI_DIN0_A,
		spi_sclk				=> SPI_CLK_A,
		spi_load_n			=> A_spi_ldn_front,
		
		spi_sdo				=> A_spi_sdo_front,
		injection1_out		=> injection1_out_A_front,
		threshold_pix_out	=> threshold_pix_out_A_front,
		threshold_low_out	=> threshold_low_out_A_front,
		threshold_high_out=> threshold_high_out_A_front,
		temp_dac_out		=> temp_dac_out_A_front,
		temp_adc_out		=> temp_adc_out_A_front
	 );		
	 
    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- data gen
    
    i_data_gen : entity work.data_generator
    port map (
        clk => qsfp_pll_clk,
        reset => not reset_n,
        enable_pix => '1',
        --enable_sc:         	in  std_logic;
        random_seed => (others => '1'),
        data_pix_generated => data_to_fifo,
        --data_sc_generated:   	out std_logic_vector(31 downto 0);
        data_pix_ready => data_to_fifo_we,
        --data_sc_ready:      	out std_logic;
        start_global_time => (others => '0')--,
              -- TODO: add some rate control
    );

    i_merger : entity work.data_merger
    port map (
        clk                     => qsfp_pll_clk,
        reset                   => not reset_n,
        fpga_ID_in              => (5=>'1',others => '0'),
        FEB_type_in             => "111010",
        state_idle              => '1',
        state_run_prepare       => '0',
        state_sync              => '0',
        state_running           => '0',
        state_terminating       => '0',
        state_link_test         => '0',
        state_sync_test         => '0',
        state_reset             => '0',
        state_out_of_DAQ        => '0',
        data_out                => qsfp_tx_data(31 downto 0),
        data_is_k               => qsfp_tx_datak(3 downto 0),
        data_in                 => data_from_fifo,
        data_in_slowcontrol     => sc_from_fifo,
        slowcontrol_fifo_empty  => sc_from_fifo_empty,
        data_fifo_empty         => '1',--data_from_fifo_empty,
        slowcontrol_read_req    => sc_from_fifo_re,
        data_read_req           => data_from_fifo_re,
        terminated              => open,
        override_data_in        => (others => '0'),
        override_data_is_k_in   => (others => '0'),
        override_req            => '0',
        override_granted        => open,
        data_priority           => '0',
        leds                    => open -- debug
    );

    i_data_fifo : entity work.mergerfifo
    generic map (
        DEVICE => "Stratix IV"--,
    )
    port map (
        data    => data_to_fifo,
        rdclk   => qsfp_pll_clk,
        rdreq   => data_from_fifo_re,
        wrclk   => qsfp_pll_clk,
        wrreq   => data_to_fifo_we,
        q       => data_from_fifo,
        rdempty => data_from_fifo_empty,
        wrfull  => open--,
    );

    i_sc_fifo : entity work.mergerfifo -- ip_fifo
    generic map (
--        ADDR_WIDTH => 11,
--        DATA_WIDTH => 36,
        DEVICE => "Stratix IV"--,
    )
    port map (
        data    => sc_to_fifo,
        rdclk   => qsfp_pll_clk,
        rdreq   => sc_from_fifo_re,
        wrclk   => qsfp_pll_clk,
        wrreq   => sc_to_fifo_we,
        q       => sc_from_fifo,
        rdempty => sc_from_fifo_empty,
        wrfull  => open--,
    );

    ----------------------------------------------------------------------------

    ----------------------------------------------------------------------------
    -- POD

--    pod_tx_reset <= '0';
--    pod_tx_reset <= '0';
--
--    i_pod : entity work.xcvr_s4
--    generic map (
--        data_rate => 5000,
--        pll_freq => 125--,
--    )
--    port map (
--        -- avalon slave interface
--        avs_address     => avm_pod.address(15 downto 2),
--        avs_read        => avm_pod.read,
--        avs_readdata    => avm_pod.readdata,
--        avs_write       => avm_pod.write,
--        avs_writedata   => avm_pod.writedata,
--        avs_waitrequest => avm_pod.waitrequest,
--
--        tx_data     => X"02CAFE" & work.util.D28_5
--                     & X"02BABE" & work.util.D28_5
--                     & X"01DEAD" & work.util.D28_5
--                     & X"00BEEF" & work.util.D28_5,
--        tx_datak    => "0001"
--                     & "0001"
--                     & "0001"
--                     & "0001",
--
--        rx_data => open,
--        rx_datak => open,
--
--        tx_clkout   => open,
--        tx_clkin    => (others => pod_pll_clk),
--        rx_clkout   => open,
--        rx_clkin    => (others => pod_pll_clk),
--
--        tx_p        => pod_tx,
--        rx_p        => pod_rx,
--
--        pll_refclk  => pod_pll_clk,
--        cdr_refclk  => pod_pll_clk,
--
--        reset   => not nios_reset_n,
--        clk     => nios_clk--,
--    );

    ----------------------------------------------------------------------------

end architecture;
