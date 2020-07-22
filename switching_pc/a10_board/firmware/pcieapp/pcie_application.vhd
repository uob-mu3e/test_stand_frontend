-----------------------------------------------------------------------------
-- PCIe application block, handles all the pcie stuff not handled by the IP core
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.pcie_components.all;
--use work.mupix_types.all;
use work.mudaq_registers.all;

entity pcie_application is
	generic (
			DMAMEMWRITEADDRSIZE : integer := 14;
			DMAMEMREADADDRSIZE  : integer := 12;
			DMAMEMWRITEWIDTH	  : integer := 32
		);
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- to IF
		tx_st_data0 :  		out 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		tx_st_eop0 :  			out		STD_LOGIC;
		tx_st_sop0 :  			out 	STD_LOGIC;
		tx_st_ready0 :			in 	STD_LOGIC;
		tx_st_valid0 :			out 	STD_LOGIC;
		tx_st_empty0 :			out 	STD_LOGIC_VECTOR(1 downto 0);
		
		-- from Config
		completer_id :			in	STD_LOGIC_VECTOR(12 downto 0);
		
		-- from IF
		rx_st_data0 :  		in 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		rx_st_eop0 :  			in		STD_LOGIC;
		rx_st_sop0 :  			in 	STD_LOGIC;
		rx_st_ready0 :			out 	STD_LOGIC;
		rx_st_valid0 :			in 	STD_LOGIC;
		rx_bar0 :				in 	STD_LOGIC_VECTOR (7 downto 0);
		
		-- Interrupt stuff
		app_msi_req:			out	std_logic;
		app_msi_tc:				out 	std_logic_vector(2 downto 0);
		app_msi_num:			out	std_logic_vector(4 downto 0);
		app_msi_ack:			in		std_logic;
		
		-- registers
		writeregs :				out	reg32array;
		regwritten :			out   std_logic_vector(63 downto 0);
		readregs :				in 	reg32array;
		
		-- pcie writeable memory
		writememclk		  :   in std_logic;
		writememreadaddr :	in std_logic_vector(15 downto 0);
		writememreaddata :	out STD_LOGIC_VECTOR (31 DOWNTO 0);
		
		-- pcie readable memory
		readmem_data 		: 	in std_logic_vector(31 downto 0);
		readmem_addr 		: 	in std_logic_vector(15 downto 0);
		readmemclk			:	in std_logic;
		readmem_wren		:  in std_logic;
		readmem_endofevent:  in std_logic;
		
		-- dma memory
		dma_data 			: 	in std_logic_vector(DMAMEMWRITEWIDTH-1 downto 0);
		dmamemclk			:	in std_logic;
		dmamem_wren			:  in std_logic;
		dmamem_endofevent	:  in std_logic;
		dmamemhalffull		:  out std_logic;
		
		-- second dma memory
		dma2_data 			: 	in std_logic_vector(DMAMEMWRITEWIDTH-1 downto 0);
		dma2memclk			:	in std_logic;
		dma2mem_wren		:  in std_logic;
		dma2mem_endofevent:  in std_logic;
		dma2memhalffull	:  out std_logic;
		
		-- test ports  
		testout				: out STD_LOGIC_VECTOR (127 DOWNTO 0);
		testin				: in  STD_LOGIC_VECTOR (127 DOWNTO 0);
		testout_ena			: out std_logic;
		pb_in					: in std_logic_vector(2 downto 0);
		inaddr32_r			: out STD_LOGIC_VECTOR (31 DOWNTO 0);
		inaddr32_w			: out STD_LOGIC_VECTOR (31 DOWNTO 0)
		
	);
end entity;



architecture RTL of pcie_application is

	
		-- from register read part
		signal rreg_readaddr :			 std_logic_vector(5 downto 0);
		signal rreg_readlength :		 std_logic_vector(9 downto 0);
		signal rreg_header2 :			 std_logic_vector(31 downto 0);
		signal rreg_readen :				 std_logic;
		
		-- from register write part
		signal wreg_readaddr :			 std_logic_vector(5 downto 0);
		signal wreg_readlength :		 std_logic_vector(9 downto 0);
		signal wreg_header2 :			 std_logic_vector(31 downto 0);
		signal wreg_readen :				 std_logic;		
		
		-- from memory read part
		signal rmem_readaddr :			 std_logic_vector(15 downto 0);
		signal rmem_readlength :		 std_logic_vector(9 downto 0);
		signal rmem_header2 :			 std_logic_vector(31 downto 0);
		signal rmem_header2_reg:		 std_logic_vector(31 downto 0);
		signal rmem_readen :				 std_logic;
		
		-- from memory write part
		signal wmem_readaddr :			 std_logic_vector(15 downto 0);
		signal wmem_readlength :		 std_logic_vector(9 downto 0);
		signal wmem_header2 :			 std_logic_vector(31 downto 0);
		signal wmem_readen :				 std_logic;
		
		signal rx_st_ready_rreg : 		std_logic;
		signal rx_st_ready_wreg : 		std_logic;
		
		signal rx_st_ready_rmem : 		std_logic;
		signal rx_st_ready_wmem : 		std_logic;
		
		-- registers
		signal writeregs_s :				reg32array;
		signal regwritten_s :			std_logic_vector(63 downto 0);
		signal readregs_s :				reg32array;
		signal readregs_int :			reg32array;
		
		signal writememaddr:				std_logic_vector(15 downto 0);
		signal writememaddr_r:			std_logic_vector(15 downto 0);
		signal writememaddr_w:			std_logic_vector(15 downto 0);
		signal writememdata:				std_logic_vector(31 downto 0);
		signal writememwren:				std_logic;
		signal writememq:					std_logic_vector(31 downto 0);
		
		
		signal readmem_readaddr : 	std_logic_vector(13 downto 0);
		signal readmem_readdata : 	std_logic_vector(127 downto 0);
				
		signal testtrig:					std_logic;
		
		
		-- dma
		signal dma_request:			std_logic;
		signal dma_granted:			std_logic;
		signal dma_done:				std_logic;
		signal dma2_request:			std_logic;
		signal dma2_granted:			std_logic;
		signal dma2_done:				std_logic;
		
		signal dma_tx_ready:			std_logic;
		signal dma_tx_data:			std_logic_vector(255 downto 0);
		signal dma_tx_valid:			std_logic;
		signal dma_tx_sop:			std_logic;
		signal dma_tx_eop:			std_logic;
		signal dma_tx_empty:			std_logic_vector(1 downto 0);
		
		signal dma2_tx_ready:		std_logic;
		signal dma2_tx_data:			std_logic_vector(255 downto 0);
		signal dma2_tx_valid:		std_logic;
		signal dma2_tx_sop:			std_logic;
		signal dma2_tx_eop:			std_logic;
		signal dma2_tx_empty:		std_logic_vector(1 downto 0);
		
		signal dma_control_address:	 		std_logic_vector(63 downto 0);
		signal dma2_control_address:	 		std_logic_vector(63 downto 0);
		signal dma_data_address:	 			std_logic_vector(63 downto 0);
		signal dma_data_address_out:			std_logic_vector(63 downto 0);
		signal dma2_data_address_out:			std_logic_vector(63 downto 0);
		signal dma_data_mem_addr:				std_logic_vector(11 downto 0);
		signal dma_data_pages:					std_logic_vector(19 downto 0);
		signal dma_data_pages_out:				std_logic_vector(19 downto 0);
		signal dma2_data_pages_out:			std_logic_vector(19 downto 0);
		signal dma_data_n_addrs:				std_logic_vector(11 downto 0);
		signal dma2_data_n_addrs:				std_logic_vector(11 downto 0);
		signal dma_write_config:				std_logic;
		signal dma2_write_config:				std_logic;
		
		signal app1_msi_req:			std_logic;
		signal app1_msi_tc:			std_logic_vector(2 downto 0);
		signal app1_msi_num:			std_logic_vector(4 downto 0);
		signal app1_msi_ack:			std_logic;
		
		signal app2_msi_req:			std_logic;
		signal app2_msi_tc:			std_logic_vector(2 downto 0);
		signal app2_msi_num:			std_logic_vector(4 downto 0);
		signal app2_msi_ack:			std_logic;
		
		signal testout_completer : 	std_logic_vector(127 downto 0);
		signal testout_dma:				std_logic_vector(71 downto 0);
		
		
		begin
		
		writeregs 	<= writeregs_s;
		regwritten	<= regwritten_s;
		readregs_s	<= readregs_int(63 downto 56) & readregs(55 downto 0);
		
		
		rx_st_ready0 <= '1' when rx_st_ready_wreg = '1' and rx_st_ready_rreg = '1' and rx_st_ready_wmem = '1' and rx_st_ready_rmem = '1'  -- should we add the DMA here somehow?
									else '0';
			
		-- needs to changed once meory is added
		--rx_st_ready_rmem <= '1';
		--rx_st_ready_wmem <= '1';
		

    e_pcie_writeable_registers : entity work.pcie_writeable_registers
		port map(
			local_rstn		=> local_rstn,
			refclk			=> refclk,
	
			-- from IF
			rx_st_data0 	=> rx_st_data0,
			rx_st_eop0   	=> rx_st_eop0,
			rx_st_sop0 		=> rx_st_sop0,
			rx_st_ready0 	=> rx_st_ready_wreg,
			rx_st_valid0 	=>	rx_st_valid0,
			rx_bar 			=> rx_bar0(0),
		
			-- registers
			writeregs 		=> writeregs_s,
			regwritten		=> regwritten_s,
		
			-- to response engine
			readaddr 		=> wreg_readaddr,
			readlength 		=> wreg_readlength,
			header2 			=>	wreg_header2,
			readen 			=> wreg_readen,
			-- debugging
			inaddr32_w		=> inaddr32_w
		);
		
		
		-- map to test port
		--process(refclk, local_rstn)
		--begin
		
		--if(local_rstn = '0') then
		--	testout <= (others => '0');
		--	testout_ena <= '0';
		--	testtrig <= '0';
		--elsif(refclk'event and refclk = '1') then
		--	if(rx_bar0(0) = '1' and rx_st_sop0 = '1') then
		--		testout 		<= rx_st_data0;
		--		testout_ena <= '1';
		--		testtrig		<= '1';
		--	end if;
		--	if(pb_in(1) = '0') then
		--		testtrig <= '0';
		--	end if;
		--end if;
		--end process;
		
		
		

    e_pcie_readable_registers : entity work.pcie_readable_registers
		port map(
			local_rstn		=> local_rstn,
			refclk			=> refclk,
	
			-- from IF
			rx_st_data0 	=> rx_st_data0,	
			rx_st_eop0   	=> rx_st_eop0,
			rx_st_sop0 		=> rx_st_sop0,
			rx_st_ready0 	=> rx_st_ready_rreg,
			rx_st_valid0 	=>	rx_st_valid0,
			rx_bar 			=> rx_bar0(1),
				
			-- to response engine
			readaddr 		=> rreg_readaddr,
			readlength 		=> rreg_readlength,
			header2 			=>	rreg_header2,
			readen 			=> rreg_readen,
			-- debugging
			inaddr32_r		=> inaddr32_r
		);


    e_pcie_writeable_memory : entity work.pcie_writeable_memory
		port map(
			local_rstn		=> local_rstn,
			refclk			=> refclk,
	
			-- from IF
			rx_st_data0 	=> rx_st_data0,
			rx_st_eop0   	=> rx_st_eop0,
			rx_st_sop0 		=> rx_st_sop0,
			rx_st_ready0 	=> rx_st_ready_wmem,
			rx_st_valid0 	=>	rx_st_valid0,
			rx_bar 			=> rx_bar0(2),
		
			-- to memory
			tomemaddr 		=> writememaddr_w,
			tomemdata 		=> writememdata,
			tomemwren 		=> writememwren,
		
			-- to response engine
			readaddr 		=> wmem_readaddr,
			readlength 		=> wmem_readlength,
			header2 			=>	wmem_header2,
			readen 			=> wmem_readen
		);


    e_pcie_readable_memory : entity work.pcie_readable_memory
		port map(
			local_rstn		=> local_rstn,
			refclk			=> refclk,
	
			-- from IF
			rx_st_data0 	=> rx_st_data0,
			rx_st_eop0   	=> rx_st_eop0,
			rx_st_sop0 		=> rx_st_sop0,
			rx_st_ready0 	=> rx_st_ready_rmem,
			rx_st_valid0 	=>	rx_st_valid0,
			rx_bar 			=> rx_bar0(3),
		
			-- to response engine
			readaddr 		=> rmem_readaddr,
			readlength 		=> rmem_readlength,
			header2 			=>	rmem_header2,
			readen 			=> rmem_readen
		);

		


    e_pcie_completer : entity work.pcie_completer
		port map(
			local_rstn		=> local_rstn,
			refclk			=> refclk,
	
			-- to IF
			tx_st_data0 	=> tx_st_data0,
			tx_st_eop0		=> tx_st_eop0,
			tx_st_sop0 		=> tx_st_sop0,
			tx_st_ready0_next 	=> tx_st_ready0,
			tx_st_valid0 	=> tx_st_valid0,
			tx_st_empty0 	=> tx_st_empty0,
		
			-- from Config
			completer_id 	=> completer_id,
		
		
			-- registers
			writeregs 		=> writeregs_s,
			readregs 		=> readregs_s,
			
			-- from register read part
			rreg_readaddr 		=> rreg_readaddr,
			rreg_readlength 	=> rreg_readlength,
			rreg_header2 		=> rreg_header2,
			rreg_readen 		=> rreg_readen,
		
			-- from register write part
			wreg_readaddr 		=> wreg_readaddr,
			wreg_readlength 	=> wreg_readlength,
			wreg_header2 		=> wreg_header2,
			wreg_readen 		=> wreg_readen,
		
			-- from memory read part
			rmem_readaddr 		=> rmem_readaddr,
			rmem_readlength	=> rmem_readlength,
			rmem_header2 		=> rmem_header2,
			rmem_readen 		=> rmem_readen,
		
			-- from memory write part
			wmem_readaddr 		=> wmem_readaddr,
			wmem_readlength	=> wmem_readlength,
			wmem_header2 		=> wmem_header2,
			wmem_readen 		=> wmem_readen,
			
			-- to and from writeable memory
			writemem_addr  	=> writememaddr_r,
			writemem_data		=> writememq,
			
			-- to and from readable memory
			readmem_addr  		=> readmem_readaddr,
			readmem_data		=> readmem_readdata,
			
			-- to and from dma engine
			dma_request			=> dma_request,
			dma_granted			=> dma_granted,
			dma_done				=> dma_done,
			dma_tx_ready		=> dma_tx_ready,
			dma_tx_data			=> dma_tx_data,
			dma_tx_valid		=> dma_tx_valid,
			dma_tx_sop			=> dma_tx_sop,
			dma_tx_eop			=> dma_tx_eop,
			dma_tx_empty		=> dma_tx_empty,
			
			-- to and from second dma engine
			dma2_request			=> dma2_request,
			dma2_granted			=> dma2_granted,
			dma2_done				=> dma2_done,
			dma2_tx_ready		=> dma2_tx_ready,
			dma2_tx_data			=> dma2_tx_data,
			dma2_tx_valid		=> dma2_tx_valid,
			dma2_tx_sop			=> dma2_tx_sop,
			dma2_tx_eop			=> dma2_tx_eop,
			dma2_tx_empty		=> dma2_tx_empty,
			
			
			-- test port
			testout				=> testout_completer,
			testout_ena			=> testout_ena
	);
	
	
		writememaddr <= writememaddr_r when writememwren = '0' else	
					writememaddr_w;

    e_pcie_wram_narrow : component work.cmp.pcie_wram_narrow
        PORT MAP
        (
                address_a        => writememaddr,
                address_b       	=> writememreadaddr, 
                clock_a          => refclk,
                clock_b         	=> writememclk,
                data_a          	=> writememdata,
                data_b           => (others => '0'),
                wren_a           => writememwren,
                wren_b          	=> '0',
                q_a              => writememq,
                q_b              => writememreaddata
        );

	  
		  

    e_pcie_ram_narrow : component work.cmp.pcie_ram_narrow
	PORT MAP
	(
		data						=> readmem_data,
		rdaddress				=> readmem_readaddr,
		rdclock					=> refclk,
		wraddress				=> readmem_addr,
		wrclock					=> readmemclk,
		wren						=> readmem_wren,
		q							=> readmem_readdata
	);

	



    e_dma_engine_1 : entity work.dma_engine
	generic map(
			MEMWRITEADDRSIZE => DMAMEMWRITEADDRSIZE,
			MEMREADADDRSIZE  => DMAMEMREADADDRSIZE,
			MEMWRITEWIDTH	  => DMAMEMWRITEWIDTH,
			IRQNUM			  => "00000",
			ENABLE_BIT		  => DMA_BIT_ENABLE,
			NOW_BIT			  => DMA_BIT_NOW,
			ENABLE_INTERRUPT_BIT => DMA_BIT_ENABLE_INTERRUPTS
	)
	
	port map(
		local_rstn				=> local_rstn,
		refclk					=> refclk,
		
		-- Stuff for DMA writing
		dataclk					=> dmamemclk,
		datain					=> dma_data,
		datawren					=> dmamem_wren,
		endofevent				=> dmamem_endofevent,
		memhalffull				=> dmamemhalffull,
		
				-- Bus and device number
		cfg_busdev				=> completer_id,
		
		-- Comunication with completer
		dma_request				=> dma_request,
		dma_granted				=> dma_granted,
		dma_done					=> dma_done,
		tx_ready					=> tx_st_ready0,
		tx_data					=> dma_tx_data,
		tx_valid					=> dma_tx_valid,
		tx_sop					=> dma_tx_sop,
		tx_eop					=> dma_tx_eop,
		tx_empty					=> dma_tx_empty,
		
		-- Interrupt stuff
		app_msi_req				=> app1_msi_req,
		app_msi_tc				=> app1_msi_tc,
		app_msi_num				=> app1_msi_num,
		app_msi_ack				=> app1_msi_ack,
		
		-- Configuration register
		dma_control_address				=> dma_control_address,
		dma_data_address					=> dma_data_address,
		dma_data_address_out				=> dma_data_address_out,
		dma_data_mem_addr					=> dma_data_mem_addr,
		dma_addrmem_data_written 		=> regwritten_s(DMA_DATA_ADDR_LOW_REGISTER_W),
		dma_data_pages						=> dma_data_pages,		
		dma_data_pages_out				=> dma_data_pages_out,
		dma_data_n_addrs					=> dma_data_n_addrs,
		
		dma_register						=> writeregs_s(DMA_REGISTER_W),
		dma_register_written				=> regwritten_s(DMA_REGISTER_W),
		dma_status_register				=> readregs_int(DMA_STATUS_REGISTER_R),
		test_out								=> testout_dma
		
		);
		
		

    e_dma_engine_2 : entity work.dma_engine
	generic map(
			MEMWRITEADDRSIZE => DMAMEMWRITEADDRSIZE,
			MEMREADADDRSIZE  => DMAMEMREADADDRSIZE,
			MEMWRITEWIDTH	  => DMAMEMWRITEWIDTH,
			IRQNUM			  => "00001",
			ENABLE_BIT		  => DMA2_BIT_ENABLE,
			NOW_BIT			  => DMA2_BIT_NOW,
			ENABLE_INTERRUPT_BIT => DMA2_BIT_ENABLE_INTERRUPTS
	)
	
	port map(
		local_rstn				=> local_rstn,
		refclk					=> refclk,
		
		-- Stuff for DMA writing
		dataclk					=> dma2memclk,
		datain					=> dma2_data,
		datawren					=> dma2mem_wren,
		endofevent				=> dma2mem_endofevent,
		memhalffull				=> dma2memhalffull,
		
				-- Bus and device number
		cfg_busdev				=> completer_id,
		
		-- Comunication with completer
		dma_request				=> dma2_request,
		dma_granted				=> dma2_granted,
		dma_done					=> dma2_done,
		tx_ready					=> tx_st_ready0,
		tx_data					=> dma2_tx_data,
		tx_valid					=> dma2_tx_valid,
		tx_sop					=> dma2_tx_sop,
		tx_eop					=> dma2_tx_eop,
		tx_empty					=> dma2_tx_empty,
		
		-- Interrupt stuff
		app_msi_req				=> app2_msi_req,
		app_msi_tc				=> app2_msi_tc,
		app_msi_num				=> app2_msi_num,
		app_msi_ack				=> app2_msi_ack,
		
		-- Configuration register
		dma_control_address				=> dma2_control_address,
		dma_data_address					=> dma_data_address,
		dma_data_address_out				=> dma2_data_address_out,
		dma_data_mem_addr					=> dma_data_mem_addr,
		dma_addrmem_data_written 		=> regwritten_s(DMA_DATA_ADDR_LOW_REGISTER_W),
		dma_data_pages						=> dma_data_pages,		
		dma_data_pages_out				=> dma2_data_pages_out,
		dma_data_n_addrs					=> dma2_data_n_addrs,
		
		dma_register						=> writeregs_s(DMA_REGISTER_W),
		dma_register_written				=> regwritten_s(DMA_REGISTER_W),
		dma_status_register				=> readregs_int(DMA2_STATUS_REGISTER_R),
		test_out								=> open
		
		);

		
		


process(refclk, local_rstn)
begin

if(local_rstn = '0') then
	testout <= (others => '0');
	dma_write_config 	<=  '0';
	dma2_write_config <=  '0';
	
	app_msi_req		<= '0';		
	app_msi_tc		<= (others => '0');		
	app_msi_num		<= (others => '0');		
	app1_msi_ack	<= '0';
	app2_msi_ack	<= '0';

elsif(refclk'event and refclk = '1') then

	-- IRQ arbitration
	if(app1_msi_req = '1' and app1_msi_ack = '0') then
		app_msi_req 	<= '1';
		app_msi_tc		<= app1_msi_tc;
		app_msi_num		<= app1_msi_num;
		app1_msi_ack	<= app_msi_ack; 
	elsif(app2_msi_req = '1' and app2_msi_ack = '0') then
		app_msi_req 	<= '1';
		app_msi_tc		<= app2_msi_tc;
		app_msi_num		<= app2_msi_num;
		app2_msi_ack	<= app_msi_ack;
	else
		app_msi_req 	<= '0';
		app1_msi_ack	<= '0';
		app2_msi_ack	<= '0';
	end if;

	if ( rmem_header2 /= "00000000000000000000000000000000" ) then	
		rmem_header2_reg <= rmem_header2;
	end if;

	dma_control_address 				<= writeregs_s(DMA_CTRL_ADDR_HI_REGISTER_W) & writeregs_s(DMA_CTRL_ADDR_LOW_REGISTER_W);
	dma2_control_address 			<= writeregs_s(DMA2_CTRL_ADDR_HI_REGISTER_W) & writeregs_s(DMA2_CTRL_ADDR_LOW_REGISTER_W);
	dma_data_address					<= writeregs_s(DMA_DATA_ADDR_HI_REGISTER_W) & writeregs_s(DMA_DATA_ADDR_LOW_REGISTER_W); 
	dma_data_n_addrs					<= writeregs_s(DMA_NUM_ADDRESSES_REGISTER_W)(DMA_NUM_ADDRESSES_RANGE);
	dma2_data_n_addrs					<= writeregs_s(DMA_NUM_ADDRESSES_REGISTER_W)(DMA2_NUM_ADDRESSES_RANGE);
	dma_data_mem_addr					<= writeregs_s(DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W)(DMA_RAM_LOCATION_RANGE);
	dma_data_pages						<= writeregs_s(DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W)(DMA_NUM_PAGES_RANGE);
	readregs_int(DMA_DATA_ADDR_HI_REGISTER_R)		<= dma_data_address_out(63 downto 32);
	readregs_int(DMA_DATA_ADDR_LOW_REGISTER_R) 	<= dma_data_address_out(31 downto 0);
	readregs_int(DMA2_DATA_ADDR_HI_REGISTER_R)		<= dma2_data_address_out(63 downto 32);
	readregs_int(DMA2_DATA_ADDR_LOW_REGISTER_R) 	<= dma2_data_address_out(31 downto 0);
	readregs_int(DMA_NUM_PAGES_REGISTER_R)(DMA_NUM_PAGES_RANGE) <= dma_data_pages_out;		
	readregs_int(DMA2_NUM_PAGES_REGISTER_R)(DMA_NUM_PAGES_RANGE) <= dma2_data_pages_out;
	
	if(regwritten_s(DMA_DATA_ADDR_LOW_REGISTER_W)='1' and writeregs_s(DMA_REGISTER_W)(DMA_BIT_ADDR_WRITE_ENABLE) = '1') then
		dma_write_config <= '1';
	else
		dma_write_config <= '0';
	end if;
	
	if(regwritten_s(DMA_DATA_ADDR_LOW_REGISTER_W)='1' and writeregs_s(DMA_REGISTER_W)(DMA2_BIT_ADDR_WRITE_ENABLE) = '1') then
		dma2_write_config <= '1';
	else
		dma2_write_config <= '0';
	end if;
	
	testout(127 downto 124)	<= testout_completer(127 downto 124);
	testout(123 downto 112) <= "00" & rmem_readlength;                        -- length of read request for readable memory
	testout(111 downto 108) <= testout_completer(123 downto 120);      -- empty of FIFOs containing read & write requests
	--testout(107 downto 52) 	<= testout_dma(55 downto 0);               
	testout(107 downto 92)  <= rmem_readaddr;
	--testout(91 downto 88)	<= "000" & rmem_readen;
	testout(87 downto 56) 	<= rmem_header2_reg;
	testout(55 downto 52 ) 	<= (others => '0');
	testout(51 downto 20) 	<= readregs_int(56);                       -- DMA status register
	testout(19 downto 8)		<= testin(11 downto 0);
	testout(7 downto 0) 		<= testout_completer(7 downto 0);
	
	--readregs_int(60)    <= (others => '0');
	--readregs_int(61)    <= (others => '0');
	--readregs_int(62)    <= (others => '0');
	--readregs_int(63)    <= (others => '0');
end if;
end process;

end architecture;
