constant spi_clk_period		: time := 10 ns;

------------------------------------------------------------------------
	-- ===SIMULATION PROCESS FOR SPI=== --{{{

	--GENERATE THE SPI CLOCK --{{{
	s_spi_clk <= not s_spi_clk after spi_clk_period/2;
	asics_sclk <= s_spi_clk when s_spi_clk_ena = '1' else '0';	--GATED CLOCK SIGNAL
	--}}}

	-- convert config to SPI data
	SPI_data <= config_to_vector(config);
	slave_config <= vector_to_config(SPI_return_data);

	sim_spi : process --{{{
		variable seed1, seed2: positive;
		-- Random real-number value in range 0 to 1.0
		variable rand: real;
		-- Random integer value in range 0=> 4095
		variable int_rand : integer;
		variable n_ser_events : integer := 3;
	begin

		seed1:=1483;
		seed2:=2356;

		s_spi_clk_ena <= '0';				--ENABLE THE OUTPUT CLOCK
		asics_mosi <= '0';
		asic_cs <=  (others => '1');

		wait for 2 us;

		--========Generate and transmit new values=============		--{{{
		-- generate random config data for channels
		-- gen_idle_signal
		config.gen_idle_signal	<= '1';
		config.recv_rec_all	<= '0';

		-- generate configurations for L1_fifo external trigger
		config.fifo_trig_mode		<= '0';
		config.fifo_trig_back_time	<= (others => '0');
		config.fifo_trig_sign_forw_time	<= '0';
		config.fifo_trig_forw_time	<= (others => '0');

		--The configuration for the master slave select
		config.ms_limits	<= "00000";
		config.ms_switch_sel	<= '0';
		config.ms_debug		<= '0';

		-- frame_generator configurations
		config.prbs_debug	<= '0';
		config.single_prbs	<= '0';
		config.fast_trans_mode	<= '1';

		-- pll configurations
		config.pll1_SetCoarse		<= '0';
		config.pll1_EnVCOMonitor	<= '0';
		config.disable_coarse		<= '0';


		-- configurations for analog channels --{{{
		for i in 0 to N_CHANNELS-1 loop

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.Anode_flag(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.Cathode_flag(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.S_switch(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.SorD(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.SorD_not(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.edge(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.edge_cml(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.DAC_cmlscale(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*4.0));
			config.comp_spi(i) <=  std_logic_vector(to_unsigned(int_rand,2));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*128.0));
			config.DAC_SiPM(i) <=  std_logic_vector(to_unsigned(int_rand,7));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*512.0));
			config.DAC_Tthresh(i) <=  std_logic_vector(to_unsigned(int_rand,9));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*256.0));
			config.DAC_ampcom(i) <=  std_logic_vector(to_unsigned(int_rand,8));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*128.0));
			config.DAC_inputbias(i) <=  std_logic_vector(to_unsigned(int_rand,7));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*256.0));
			config.DAC_Ethresh(i) <=  std_logic_vector(to_unsigned(int_rand,8));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*128.0));
			config.DAC_pole(i) <=  std_logic_vector(to_unsigned(int_rand,7));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*16.0));
			config.DAC_cml(i) <=  std_logic_vector(to_unsigned(int_rand,4));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*4.0));
			config.DAC_delay(i) <=  std_logic_vector(to_unsigned(int_rand,2));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*8.0));
			config.amon_ctrl(i) <=  std_logic_vector(to_unsigned(int_rand,3));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.dmon_ena(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.dmon_sw(i) <=  std_logic(to_unsigned(int_rand,1)(0));

			UNIFORM(seed1, seed2, rand);
			int_rand := INTEGER(TRUNC(rand*1.0));
			config.tdctest(i) <=  std_logic(to_unsigned(int_rand,1)(0));
		end loop;
		--}}}

		-- channel mask
		config.channel_mask	<= (others => '0');

		-- TDC bias DAC and monitor DACs --{{{
		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*65536.0));
		config.dac_tdc_bias(0 to 15)	<=  std_logic_vector(to_unsigned(int_rand,16));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*65536.0));
		config.dac_tdc_bias(16 to 31)	<=  std_logic_vector(to_unsigned(int_rand,16));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*65536.0));
		config.dac_tdc_bias(32 to 47)	<=  std_logic_vector(to_unsigned(int_rand,16));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*65536.0));
		config.dac_tdc_bias(48 to 63)	<=  std_logic_vector(to_unsigned(int_rand,16));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*256.0));
		config.dac_tdc_bias(64 to 71)	<=  std_logic_vector(to_unsigned(int_rand,8));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*4096.0));
		config.dac_tdc_latchbias	<=  std_logic_vector(to_unsigned(int_rand,12));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*1.0));
		config.amon_en		<=  std_logic(to_unsigned(int_rand,1)(0));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*256.0));
		config.amon_DAC		<=  std_logic_vector(to_unsigned(int_rand,8));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*1.0));
		config.dig_mon1_en	<=  std_logic(to_unsigned(int_rand,1)(0));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*256.0));
		config.dig_mon1_DAC	<=  std_logic_vector(to_unsigned(int_rand,8));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*1.0));
		config.dig_mon2_en	<=  std_logic(to_unsigned(int_rand,1)(0));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*256.0));
		config.dig_mon2_DAC	<=  std_logic_vector(to_unsigned(int_rand,8));

		UNIFORM(seed1, seed2, rand);
		int_rand := INTEGER(TRUNC(rand*16384.0));
		config.txd_lvds_tx_dac	<=  std_logic_vector(to_unsigned(int_rand,14));
		--}}}

		--}}}


		--==============TRANSMIT THE DATA VIA SPI=============		--{{{

		asic_cs(i) <= '0';							--SELECT THE CHIP
		wait until falling_edge(s_spi_clk);
		s_spi_clk_ena <= '1';						--ENABLE THE OUTPUT CLOCK
		for i in 0 to N_CONF_BITS -1 loop
			asics_mosi <= SPI_data((N_CONF_BITS - 1) - i);					--SET THE NEXT BIT ON THE asics_mosi LINE
			-- for other simulations than back-annotated simulation
			--SPI_return_data <= asics_miso & SPI_return_data(0 to SPI_return_data'high -1);	--SAMPLE THE DATA ON THE MASTER INPUT LINE
			wait until falling_edge(s_spi_clk);
			---- in some case of back-annotated simulation (sdfmax),
			---- there is a long delay between the output of spi module and the sdo of the chip, more delay is requited here.
			---- thus, the SPI_return_data should take the data 1 clk cycle later
			SPI_return_data <= asics_miso & SPI_return_data(0 to SPI_return_data'high -1);	--SAMPLE THE DATA ON THE MASTER INPUT LINE
		end loop;
		s_spi_clk_ena <= '0';						--DISABLE THE SERIAL CLOCK AGAIN
		asic_cs(i) <= '1';

		wait for 5 us;							--SECOND SENDING TO READ THE DATA BACK FROM THE SLAVE DEVICE
		asic_cs(i) <= '0';
		wait until falling_edge(s_spi_clk);
		s_spi_clk_ena <= '1';
		for i in 0 to N_CONF_BITS-1 loop
			asics_mosi <= SPI_data((N_CONF_BITS - 1) -i);					--SET THE NEXT BIT ON THE asics_mosi LINE
			-- for other simulations than back-annotated simulation
			--SPI_return_data <= asics_miso & SPI_return_data(0 to SPI_return_data'high -1);	--SAMPLE THE DATA ON THE MASTER INPUT LINE
			wait until falling_edge(s_spi_clk);
			---- in back-annotated simulation, there is a long delay between the output of spi module and the sdo of the chip, more delay is requited here.
			---- thus, the SPI_return_data should take the data 1 clk cycle later
			SPI_return_data <= asics_miso & SPI_return_data(0 to SPI_return_data'high -1);	--SAMPLE THE DATA ON THE MASTER INPUT LINE
		end loop;
		s_spi_clk_ena <= '0';						--DISABLE THE SERIAL CLOCK AGAIN
		asic_cs(i) <= '1';							--DESELECT THE CHIP AND LATCH THE SENT VALUES TO THE KLAUS OUTPUTS

		wait for 50 ns; 							--NEEDED TO FORCE asic_cs(i) = '1' IN THE SIMULATION PROCESS OTHERWISE WARNING FROM POWERVALUE
		--}}}

		for i in 0 to N_CONF_BITS -1 loop
			assert (SPI_data(i) = '0' or SPI_data(i) = '1') report "** SPI slow control: SPI_data bit " & integer'image(i) & " are not '1' or '0'" severity warning;
			assert (SPI_return_data(i) = '0' or SPI_return_data(i) = '1') report "** SPI slow control: SPI_return_data bit " & integer'image(i) & " are not '1' or '0'" severity warning;
		end loop;

		assert (SPI_return_data = SPI_data) report "** SPI slow control: data transmission failure" severity error;

		assert (SPI_return_data = config_to_vector(slave_config) ) report "** SPI slow control: inconsistency between output values and configuration" severity error;
		wait;
	end process sim_spi;
	--}}}

	--}}}


