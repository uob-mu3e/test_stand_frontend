library ieee;
use ieee.std_logic_1164.all;

package pcie_components is

subtype reg32           is std_logic_vector(31 downto 0);
type reg32array is array (63 downto 0) of reg32;

component pcie_block
generic (
			DMAMEMWRITEADDRSIZE : integer := 14;
			DMAMEMREADADDRSIZE  : integer := 12;
			DMAMEMWRITEWIDTH	  : integer := 32
);
port (
		local_rstn:				in		std_logic;
		appl_rstn:				in 	std_logic;
		refclk:					in		std_logic;
		pcie_fastclk_out:		out	std_logic; -- 250 MHz clock
		--//PCI-Express--------------------------//25 pins //--------------------------
		pcie_rx_p: 				in 		std_logic_vector(7 downto 0);           --//PCIe Receive Data-req's OCT
		pcie_tx_p: 				out		std_logic_vector(7 downto 0);           --//PCIe Transmit Data
		pcie_refclk_p:			in 		std_logic;       						--//PCIe Clock- Terminate on MB
		pcie_led_g2: 			out 	std_logic;         						--//User LED - Labeled Gen2
		pcie_led_x1: 			out 	std_logic;         						--//User LED - Labeled x1
		pcie_led_x4: 			out 	std_logic;         						--//User LED - Labeled x4
		pcie_led_x8: 			out 	std_logic;         						--//User LED - Labeled x8
		pcie_perstn: 			in 		std_logic;         						--//PCIe Reset 
		pcie_smbclk: 			in 		std_logic;         						--//SMBus Clock (TR=0)
		pcie_smbdat:			in 	std_logic;         						--//SMBus Data (TR=0)
		pcie_waken: 			out 	std_logic;          						--//PCIe Wake-Up (TR=0)	
		
		-- LEDs
		alive_led:				out		std_logic;
		comp_led:				out		std_logic;
		L0_led:					out		std_logic;
		
				-- pcie registers
		writeregs:				out		reg32array;
		regwritten:				out		std_logic_vector(63 downto 0);
		readregs:				in			reg32array;
	
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
		testout_ena			: out std_logic;
		pb_in					: in std_logic_vector(2 downto 0);
		inaddr32_r			: out STD_LOGIC_VECTOR (31 DOWNTO 0);
		inaddr32_w			: out STD_LOGIC_VECTOR (31 DOWNTO 0)
);
end component;


component pcie_writeable_registers is 
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
		rx_st_data0 :  		in 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		rx_st_eop0 :  			in		STD_LOGIC;
		rx_st_sop0 :  			in 	STD_LOGIC;
		rx_st_ready0 :			out 	STD_LOGIC;
		rx_st_valid0 :			in 	STD_LOGIC;
		rx_bar :					in 	STD_LOGIC;
		
		-- registers
		writeregs :				out	reg32array;
		regwritten :			out   std_logic_vector(63 downto 0);
		
		-- to response engine
		readaddr :				out std_logic_vector(5 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic;
		inaddr32_w			: out STD_LOGIC_VECTOR (31 DOWNTO 0)
	
	);
end component;

component pcie_readable_registers is 
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
		rx_st_data0 :  		in 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		rx_st_eop0 :  			in		STD_LOGIC;
		rx_st_sop0 :  			in 	STD_LOGIC;
		rx_st_ready0 :			out 	STD_LOGIC;
		rx_st_valid0 :			in 	STD_LOGIC;
		rx_bar :					in 	STD_LOGIC;
				
		-- to response engine
		readaddr :				out std_logic_vector(5 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic;
		inaddr32_r			: out STD_LOGIC_VECTOR (31 DOWNTO 0)
	
	);
end component;


component pcie_writeable_memory is 
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
		rx_st_data0 :  		in 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		rx_st_eop0 :  			in		STD_LOGIC;
		rx_st_sop0 :  			in 	STD_LOGIC;
		rx_st_ready0 :			out 	STD_LOGIC;
		rx_st_valid0 :			in 	STD_LOGIC;
		rx_bar :					in 	STD_LOGIC;
		
		-- to memory
		tomemaddr :				out	std_logic_vector(15 downto 0);
		tomemdata :				out	std_logic_vector(31 downto 0);
		tomemwren :				out	std_logic;
		
		-- to response engine
		readaddr :				out std_logic_vector(15 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic
	
	);
end component;

component pcie_readable_memory is 
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
		rx_st_data0 :  		in 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		rx_st_eop0 :  			in		STD_LOGIC;
		rx_st_sop0 :  			in 	STD_LOGIC;
		rx_st_ready0 :			out 	STD_LOGIC;
		rx_st_valid0 :			in 	STD_LOGIC;
		rx_bar :					in 	STD_LOGIC;
		
		-- to response engine
		readaddr :				out std_logic_vector(15 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic
	
	);
end component;


component pcie_completion_bytecount 
	port (
		fdw_be:	in std_logic_vector(3 downto 0);
		ldw_be:	in std_logic_vector(3 downto 0);	
		plength: in std_logic_vector(9 downto 0);	
		bytecount: out std_logic_vector(11 downto 0);
		lower_address : out std_logic_vector(1 downto 0)
	);
end component;


component pcie_completer is 
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
		tx_st_data0 :  		out 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		tx_st_eop0 :  			out		STD_LOGIC;
		tx_st_sop0 :  			out 	STD_LOGIC;
		tx_st_ready0_next :	in 	STD_LOGIC;
		tx_st_valid0 :			out 	STD_LOGIC;
		tx_st_empty0 :			out 	STD_LOGIC_VECTOR(1 downto 0);
		
		-- from Config
		completer_id :			in	STD_LOGIC_VECTOR(12 downto 0);
		
		
		-- registers
		writeregs :				in	reg32array;
		readregs :				in reg32array;
		
		-- from register read part
		rreg_readaddr :			in std_logic_vector(5 downto 0);
		rreg_readlength :			in std_logic_vector(9 downto 0);
		rreg_header2 :				in std_logic_vector(31 downto 0);
		rreg_readen :				in std_logic;
		
		-- from register write part
		wreg_readaddr :			in std_logic_vector(5 downto 0);
		wreg_readlength :			in std_logic_vector(9 downto 0);
		wreg_header2 :				in std_logic_vector(31 downto 0);
		wreg_readen :				in std_logic;		
		
		-- from memory read part
		rmem_readaddr :			in std_logic_vector(15 downto 0);
		rmem_readlength :			in std_logic_vector(9 downto 0);
		rmem_header2 :				in std_logic_vector(31 downto 0);
		rmem_readen :				in std_logic;
		
		-- from memory write part
		wmem_readaddr :			in std_logic_vector(15 downto 0);
		wmem_readlength :			in std_logic_vector(9 downto 0);
		wmem_header2 :				in std_logic_vector(31 downto 0);
		wmem_readen :				in std_logic;
		
		-- to and from writeable memory
		writemem_addr  : 			out std_logic_vector(15 downto 0);
		writemem_data	:        in std_logic_vector(31 downto 0);
		
		-- to and from readable memory
		readmem_addr  : 			out std_logic_vector(13 downto 0);
		readmem_data	:        in std_logic_vector(127 downto 0);
		
		-- to and from dma engine
		dma_request:			in		std_logic;
		dma_granted:			out	std_logic;
		dma_done:				in		std_logic;
		dma_tx_ready:			out	std_logic;
		dma_tx_data:			in		std_logic_vector(255 downto 0);
		dma_tx_valid:			in		std_logic;
		dma_tx_sop:				in		std_logic;
		dma_tx_eop:				in		std_logic;
		dma_tx_empty:			in		std_logic_vector(1 downto 0);
				
		-- to and from second dma engine
		dma2_request:			in		std_logic;
		dma2_granted:			out	std_logic;
		dma2_done:				in		std_logic;
		dma2_tx_ready:			out	std_logic;
		dma2_tx_data:			in		std_logic_vector(255 downto 0);
		dma2_tx_valid:			in		std_logic;
		dma2_tx_sop:			in		std_logic;
		dma2_tx_eop:			in		std_logic;
		dma2_tx_empty:			in		std_logic_vector(1 downto 0);		
		
				-- test ports  
		testout				: out STD_LOGIC_VECTOR (127 DOWNTO 0);
		testout_ena			: out std_logic
	);
	end component;
	
	
	component pcie_application is 
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
		dmamemhalffull		:  out	std_logic;
		
				-- second dma memory
		dma2_data 			: 	in std_logic_vector(DMAMEMWRITEWIDTH-1 downto 0);
		dma2memclk			:	in std_logic;
		dma2mem_wren		:  in std_logic;
		dma2mem_endofevent:  in std_logic;
		dma2memhalffull	:  out std_logic;
		
				-- test ports  
		testout				: out STD_LOGIC_VECTOR (127 DOWNTO 0);
		testin				: in STD_LOGIC_VECTOR (127 DOWNTO 0);
		testout_ena			: out std_logic;
		pb_in					: in std_logic_vector(2 downto 0);
		inaddr32_r			: out STD_LOGIC_VECTOR (31 DOWNTO 0);
		inaddr32_w			: out STD_LOGIC_VECTOR (31 DOWNTO 0)
	);
	end component;





component pcie is
	port (
		clr_st              : out std_logic;                                         --         clr_st.reset
		hpg_ctrler          : in  std_logic_vector(4 downto 0)   := (others => '0'); --      config_tl.hpg_ctrler
		tl_cfg_add          : out std_logic_vector(3 downto 0);                      --               .tl_cfg_add
		tl_cfg_ctl          : out std_logic_vector(31 downto 0);                     --               .tl_cfg_ctl
		tl_cfg_sts          : out std_logic_vector(52 downto 0);                     --               .tl_cfg_sts
		cpl_err             : in  std_logic_vector(6 downto 0)   := (others => '0'); --               .cpl_err
		cpl_pending         : in  std_logic                      := '0';             --               .cpl_pending
		coreclkout_hip      : out std_logic;                                         -- coreclkout_hip.clk
		currentspeed        : out std_logic_vector(1 downto 0);                      --   currentspeed.currentspeed
		test_in             : in  std_logic_vector(31 downto 0)  := (others => '0'); --       hip_ctrl.test_in
		simu_mode_pipe      : in  std_logic                      := '0';             --               .simu_mode_pipe
		sim_pipe_pclk_in    : in  std_logic                      := '0';             --       hip_pipe.sim_pipe_pclk_in
		sim_pipe_rate       : out std_logic_vector(1 downto 0);                      --               .sim_pipe_rate
		sim_ltssmstate      : out std_logic_vector(4 downto 0);                      --               .sim_ltssmstate
		eidleinfersel0      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel0
		eidleinfersel1      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel1
		eidleinfersel2      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel2
		eidleinfersel3      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel3
		eidleinfersel4      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel4
		eidleinfersel5      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel5
		eidleinfersel6      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel6
		eidleinfersel7      : out std_logic_vector(2 downto 0);                      --               .eidleinfersel7
		powerdown0          : out std_logic_vector(1 downto 0);                      --               .powerdown0
		powerdown1          : out std_logic_vector(1 downto 0);                      --               .powerdown1
		powerdown2          : out std_logic_vector(1 downto 0);                      --               .powerdown2
		powerdown3          : out std_logic_vector(1 downto 0);                      --               .powerdown3
		powerdown4          : out std_logic_vector(1 downto 0);                      --               .powerdown4
		powerdown5          : out std_logic_vector(1 downto 0);                      --               .powerdown5
		powerdown6          : out std_logic_vector(1 downto 0);                      --               .powerdown6
		powerdown7          : out std_logic_vector(1 downto 0);                      --               .powerdown7
		rxpolarity0         : out std_logic;                                         --               .rxpolarity0
		rxpolarity1         : out std_logic;                                         --               .rxpolarity1
		rxpolarity2         : out std_logic;                                         --               .rxpolarity2
		rxpolarity3         : out std_logic;                                         --               .rxpolarity3
		rxpolarity4         : out std_logic;                                         --               .rxpolarity4
		rxpolarity5         : out std_logic;                                         --               .rxpolarity5
		rxpolarity6         : out std_logic;                                         --               .rxpolarity6
		rxpolarity7         : out std_logic;                                         --               .rxpolarity7
		txcompl0            : out std_logic;                                         --               .txcompl0
		txcompl1            : out std_logic;                                         --               .txcompl1
		txcompl2            : out std_logic;                                         --               .txcompl2
		txcompl3            : out std_logic;                                         --               .txcompl3
		txcompl4            : out std_logic;                                         --               .txcompl4
		txcompl5            : out std_logic;                                         --               .txcompl5
		txcompl6            : out std_logic;                                         --               .txcompl6
		txcompl7            : out std_logic;                                         --               .txcompl7
		txdata0             : out std_logic_vector(31 downto 0);                     --               .txdata0
		txdata1             : out std_logic_vector(31 downto 0);                     --               .txdata1
		txdata2             : out std_logic_vector(31 downto 0);                     --               .txdata2
		txdata3             : out std_logic_vector(31 downto 0);                     --               .txdata3
		txdata4             : out std_logic_vector(31 downto 0);                     --               .txdata4
		txdata5             : out std_logic_vector(31 downto 0);                     --               .txdata5
		txdata6             : out std_logic_vector(31 downto 0);                     --               .txdata6
		txdata7             : out std_logic_vector(31 downto 0);                     --               .txdata7
		txdatak0            : out std_logic_vector(3 downto 0);                      --               .txdatak0
		txdatak1            : out std_logic_vector(3 downto 0);                      --               .txdatak1
		txdatak2            : out std_logic_vector(3 downto 0);                      --               .txdatak2
		txdatak3            : out std_logic_vector(3 downto 0);                      --               .txdatak3
		txdatak4            : out std_logic_vector(3 downto 0);                      --               .txdatak4
		txdatak5            : out std_logic_vector(3 downto 0);                      --               .txdatak5
		txdatak6            : out std_logic_vector(3 downto 0);                      --               .txdatak6
		txdatak7            : out std_logic_vector(3 downto 0);                      --               .txdatak7
		txdetectrx0         : out std_logic;                                         --               .txdetectrx0
		txdetectrx1         : out std_logic;                                         --               .txdetectrx1
		txdetectrx2         : out std_logic;                                         --               .txdetectrx2
		txdetectrx3         : out std_logic;                                         --               .txdetectrx3
		txdetectrx4         : out std_logic;                                         --               .txdetectrx4
		txdetectrx5         : out std_logic;                                         --               .txdetectrx5
		txdetectrx6         : out std_logic;                                         --               .txdetectrx6
		txdetectrx7         : out std_logic;                                         --               .txdetectrx7
		txelecidle0         : out std_logic;                                         --               .txelecidle0
		txelecidle1         : out std_logic;                                         --               .txelecidle1
		txelecidle2         : out std_logic;                                         --               .txelecidle2
		txelecidle3         : out std_logic;                                         --               .txelecidle3
		txelecidle4         : out std_logic;                                         --               .txelecidle4
		txelecidle5         : out std_logic;                                         --               .txelecidle5
		txelecidle6         : out std_logic;                                         --               .txelecidle6
		txelecidle7         : out std_logic;                                         --               .txelecidle7
		txdeemph0           : out std_logic;                                         --               .txdeemph0
		txdeemph1           : out std_logic;                                         --               .txdeemph1
		txdeemph2           : out std_logic;                                         --               .txdeemph2
		txdeemph3           : out std_logic;                                         --               .txdeemph3
		txdeemph4           : out std_logic;                                         --               .txdeemph4
		txdeemph5           : out std_logic;                                         --               .txdeemph5
		txdeemph6           : out std_logic;                                         --               .txdeemph6
		txdeemph7           : out std_logic;                                         --               .txdeemph7
		txmargin0           : out std_logic_vector(2 downto 0);                      --               .txmargin0
		txmargin1           : out std_logic_vector(2 downto 0);                      --               .txmargin1
		txmargin2           : out std_logic_vector(2 downto 0);                      --               .txmargin2
		txmargin3           : out std_logic_vector(2 downto 0);                      --               .txmargin3
		txmargin4           : out std_logic_vector(2 downto 0);                      --               .txmargin4
		txmargin5           : out std_logic_vector(2 downto 0);                      --               .txmargin5
		txmargin6           : out std_logic_vector(2 downto 0);                      --               .txmargin6
		txmargin7           : out std_logic_vector(2 downto 0);                      --               .txmargin7
		txswing0            : out std_logic;                                         --               .txswing0
		txswing1            : out std_logic;                                         --               .txswing1
		txswing2            : out std_logic;                                         --               .txswing2
		txswing3            : out std_logic;                                         --               .txswing3
		txswing4            : out std_logic;                                         --               .txswing4
		txswing5            : out std_logic;                                         --               .txswing5
		txswing6            : out std_logic;                                         --               .txswing6
		txswing7            : out std_logic;                                         --               .txswing7
		phystatus0          : in  std_logic                      := '0';             --               .phystatus0
		phystatus1          : in  std_logic                      := '0';             --               .phystatus1
		phystatus2          : in  std_logic                      := '0';             --               .phystatus2
		phystatus3          : in  std_logic                      := '0';             --               .phystatus3
		phystatus4          : in  std_logic                      := '0';             --               .phystatus4
		phystatus5          : in  std_logic                      := '0';             --               .phystatus5
		phystatus6          : in  std_logic                      := '0';             --               .phystatus6
		phystatus7          : in  std_logic                      := '0';             --               .phystatus7
		rxdata0             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata0
		rxdata1             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata1
		rxdata2             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata2
		rxdata3             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata3
		rxdata4             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata4
		rxdata5             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata5
		rxdata6             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata6
		rxdata7             : in  std_logic_vector(31 downto 0)  := (others => '0'); --               .rxdata7
		rxdatak0            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak0
		rxdatak1            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak1
		rxdatak2            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak2
		rxdatak3            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak3
		rxdatak4            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak4
		rxdatak5            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak5
		rxdatak6            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak6
		rxdatak7            : in  std_logic_vector(3 downto 0)   := (others => '0'); --               .rxdatak7
		rxelecidle0         : in  std_logic                      := '0';             --               .rxelecidle0
		rxelecidle1         : in  std_logic                      := '0';             --               .rxelecidle1
		rxelecidle2         : in  std_logic                      := '0';             --               .rxelecidle2
		rxelecidle3         : in  std_logic                      := '0';             --               .rxelecidle3
		rxelecidle4         : in  std_logic                      := '0';             --               .rxelecidle4
		rxelecidle5         : in  std_logic                      := '0';             --               .rxelecidle5
		rxelecidle6         : in  std_logic                      := '0';             --               .rxelecidle6
		rxelecidle7         : in  std_logic                      := '0';             --               .rxelecidle7
		rxstatus0           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus0
		rxstatus1           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus1
		rxstatus2           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus2
		rxstatus3           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus3
		rxstatus4           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus4
		rxstatus5           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus5
		rxstatus6           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus6
		rxstatus7           : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .rxstatus7
		rxvalid0            : in  std_logic                      := '0';             --               .rxvalid0
		rxvalid1            : in  std_logic                      := '0';             --               .rxvalid1
		rxvalid2            : in  std_logic                      := '0';             --               .rxvalid2
		rxvalid3            : in  std_logic                      := '0';             --               .rxvalid3
		rxvalid4            : in  std_logic                      := '0';             --               .rxvalid4
		rxvalid5            : in  std_logic                      := '0';             --               .rxvalid5
		rxvalid6            : in  std_logic                      := '0';             --               .rxvalid6
		rxvalid7            : in  std_logic                      := '0';             --               .rxvalid7
		rxdataskip0         : in  std_logic                      := '0';             --               .rxdataskip0
		rxdataskip1         : in  std_logic                      := '0';             --               .rxdataskip1
		rxdataskip2         : in  std_logic                      := '0';             --               .rxdataskip2
		rxdataskip3         : in  std_logic                      := '0';             --               .rxdataskip3
		rxdataskip4         : in  std_logic                      := '0';             --               .rxdataskip4
		rxdataskip5         : in  std_logic                      := '0';             --               .rxdataskip5
		rxdataskip6         : in  std_logic                      := '0';             --               .rxdataskip6
		rxdataskip7         : in  std_logic                      := '0';             --               .rxdataskip7
		rxblkst0            : in  std_logic                      := '0';             --               .rxblkst0
		rxblkst1            : in  std_logic                      := '0';             --               .rxblkst1
		rxblkst2            : in  std_logic                      := '0';             --               .rxblkst2
		rxblkst3            : in  std_logic                      := '0';             --               .rxblkst3
		rxblkst4            : in  std_logic                      := '0';             --               .rxblkst4
		rxblkst5            : in  std_logic                      := '0';             --               .rxblkst5
		rxblkst6            : in  std_logic                      := '0';             --               .rxblkst6
		rxblkst7            : in  std_logic                      := '0';             --               .rxblkst7
		rxsynchd0           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd0
		rxsynchd1           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd1
		rxsynchd2           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd2
		rxsynchd3           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd3
		rxsynchd4           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd4
		rxsynchd5           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd5
		rxsynchd6           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd6
		rxsynchd7           : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .rxsynchd7
		currentcoeff0       : out std_logic_vector(17 downto 0);                     --               .currentcoeff0
		currentcoeff1       : out std_logic_vector(17 downto 0);                     --               .currentcoeff1
		currentcoeff2       : out std_logic_vector(17 downto 0);                     --               .currentcoeff2
		currentcoeff3       : out std_logic_vector(17 downto 0);                     --               .currentcoeff3
		currentcoeff4       : out std_logic_vector(17 downto 0);                     --               .currentcoeff4
		currentcoeff5       : out std_logic_vector(17 downto 0);                     --               .currentcoeff5
		currentcoeff6       : out std_logic_vector(17 downto 0);                     --               .currentcoeff6
		currentcoeff7       : out std_logic_vector(17 downto 0);                     --               .currentcoeff7
		currentrxpreset0    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset0
		currentrxpreset1    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset1
		currentrxpreset2    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset2
		currentrxpreset3    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset3
		currentrxpreset4    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset4
		currentrxpreset5    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset5
		currentrxpreset6    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset6
		currentrxpreset7    : out std_logic_vector(2 downto 0);                      --               .currentrxpreset7
		txsynchd0           : out std_logic_vector(1 downto 0);                      --               .txsynchd0
		txsynchd1           : out std_logic_vector(1 downto 0);                      --               .txsynchd1
		txsynchd2           : out std_logic_vector(1 downto 0);                      --               .txsynchd2
		txsynchd3           : out std_logic_vector(1 downto 0);                      --               .txsynchd3
		txsynchd4           : out std_logic_vector(1 downto 0);                      --               .txsynchd4
		txsynchd5           : out std_logic_vector(1 downto 0);                      --               .txsynchd5
		txsynchd6           : out std_logic_vector(1 downto 0);                      --               .txsynchd6
		txsynchd7           : out std_logic_vector(1 downto 0);                      --               .txsynchd7
		txblkst0            : out std_logic;                                         --               .txblkst0
		txblkst1            : out std_logic;                                         --               .txblkst1
		txblkst2            : out std_logic;                                         --               .txblkst2
		txblkst3            : out std_logic;                                         --               .txblkst3
		txblkst4            : out std_logic;                                         --               .txblkst4
		txblkst5            : out std_logic;                                         --               .txblkst5
		txblkst6            : out std_logic;                                         --               .txblkst6
		txblkst7            : out std_logic;                                         --               .txblkst7
		txdataskip0         : out std_logic;                                         --               .txdataskip0
		txdataskip1         : out std_logic;                                         --               .txdataskip1
		txdataskip2         : out std_logic;                                         --               .txdataskip2
		txdataskip3         : out std_logic;                                         --               .txdataskip3
		txdataskip4         : out std_logic;                                         --               .txdataskip4
		txdataskip5         : out std_logic;                                         --               .txdataskip5
		txdataskip6         : out std_logic;                                         --               .txdataskip6
		txdataskip7         : out std_logic;                                         --               .txdataskip7
		rate0               : out std_logic_vector(1 downto 0);                      --               .rate0
		rate1               : out std_logic_vector(1 downto 0);                      --               .rate1
		rate2               : out std_logic_vector(1 downto 0);                      --               .rate2
		rate3               : out std_logic_vector(1 downto 0);                      --               .rate3
		rate4               : out std_logic_vector(1 downto 0);                      --               .rate4
		rate5               : out std_logic_vector(1 downto 0);                      --               .rate5
		rate6               : out std_logic_vector(1 downto 0);                      --               .rate6
		rate7               : out std_logic_vector(1 downto 0);                      --               .rate7
		pld_core_ready      : in  std_logic                      := '0';             --        hip_rst.pld_core_ready
		pld_clk_inuse       : out std_logic;                                         --               .pld_clk_inuse
		serdes_pll_locked   : out std_logic;                                         --               .serdes_pll_locked
		reset_status        : out std_logic;                                         --               .reset_status
		testin_zero         : out std_logic;                                         --               .testin_zero
		rx_in0              : in  std_logic                      := '0';             --     hip_serial.rx_in0
		rx_in1              : in  std_logic                      := '0';             --               .rx_in1
		rx_in2              : in  std_logic                      := '0';             --               .rx_in2
		rx_in3              : in  std_logic                      := '0';             --               .rx_in3
		rx_in4              : in  std_logic                      := '0';             --               .rx_in4
		rx_in5              : in  std_logic                      := '0';             --               .rx_in5
		rx_in6              : in  std_logic                      := '0';             --               .rx_in6
		rx_in7              : in  std_logic                      := '0';             --               .rx_in7
		tx_out0             : out std_logic;                                         --               .tx_out0
		tx_out1             : out std_logic;                                         --               .tx_out1
		tx_out2             : out std_logic;                                         --               .tx_out2
		tx_out3             : out std_logic;                                         --               .tx_out3
		tx_out4             : out std_logic;                                         --               .tx_out4
		tx_out5             : out std_logic;                                         --               .tx_out5
		tx_out6             : out std_logic;                                         --               .tx_out6
		tx_out7             : out std_logic;                                         --               .tx_out7
		derr_cor_ext_rcv    : out std_logic;                                         --     hip_status.derr_cor_ext_rcv
		derr_cor_ext_rpl    : out std_logic;                                         --               .derr_cor_ext_rpl
		derr_rpl            : out std_logic;                                         --               .derr_rpl
		dlup                : out std_logic;                                         --               .dlup
		dlup_exit           : out std_logic;                                         --               .dlup_exit
		ev128ns             : out std_logic;                                         --               .ev128ns
		ev1us               : out std_logic;                                         --               .ev1us
		hotrst_exit         : out std_logic;                                         --               .hotrst_exit
		int_status          : out std_logic_vector(3 downto 0);                      --               .int_status
		l2_exit             : out std_logic;                                         --               .l2_exit
		lane_act            : out std_logic_vector(3 downto 0);                      --               .lane_act
		ltssmstate          : out std_logic_vector(4 downto 0);                      --               .ltssmstate
		rx_par_err          : out std_logic;                                         --               .rx_par_err
		tx_par_err          : out std_logic_vector(1 downto 0);                      --               .tx_par_err
		cfg_par_err         : out std_logic;                                         --               .cfg_par_err
		ko_cpl_spc_header   : out std_logic_vector(7 downto 0);                      --               .ko_cpl_spc_header
		ko_cpl_spc_data     : out std_logic_vector(11 downto 0);                     --               .ko_cpl_spc_data
		app_int_sts         : in  std_logic                      := '0';             --        int_msi.app_int_sts
		app_int_ack         : out std_logic;                                         --               .app_int_ack
		app_msi_num         : in  std_logic_vector(4 downto 0)   := (others => '0'); --               .app_msi_num
		app_msi_req         : in  std_logic                      := '0';             --               .app_msi_req
		app_msi_tc          : in  std_logic_vector(2 downto 0)   := (others => '0'); --               .app_msi_tc
		app_msi_ack         : out std_logic;                                         --               .app_msi_ack
		npor                : in  std_logic                      := '0';             --           npor.npor
		pin_perst           : in  std_logic                      := '0';             --               .pin_perst
		pld_clk             : in  std_logic                      := '0';             --        pld_clk.clk
		pm_auxpwr           : in  std_logic                      := '0';             --     power_mgnt.pm_auxpwr
		pm_data             : in  std_logic_vector(9 downto 0)   := (others => '0'); --               .pm_data
		pme_to_cr           : in  std_logic                      := '0';             --               .pme_to_cr
		pm_event            : in  std_logic                      := '0';             --               .pm_event
		pme_to_sr           : out std_logic;                                         --               .pme_to_sr
		refclk              : in  std_logic                      := '0';             --         refclk.clk
		rx_st_bar           : out std_logic_vector(7 downto 0);                      --         rx_bar.rx_st_bar
		rx_st_mask          : in  std_logic                      := '0';             --               .rx_st_mask
		rx_st_sop           : out std_logic_vector(0 downto 0);                      --          rx_st.startofpacket
		rx_st_eop           : out std_logic_vector(0 downto 0);                      --               .endofpacket
		rx_st_err           : out std_logic_vector(0 downto 0);                      --               .error
		rx_st_valid         : out std_logic_vector(0 downto 0);                      --               .valid
		rx_st_ready         : in  std_logic                      := '0';             --               .ready
		rx_st_data          : out std_logic_vector(255 downto 0);                    --               .data
		rx_st_empty         : out std_logic_vector(1 downto 0);                      --               .empty
		tx_cred_data_fc     : out std_logic_vector(11 downto 0);                     --        tx_cred.tx_cred_data_fc
		tx_cred_fc_hip_cons : out std_logic_vector(5 downto 0);                      --               .tx_cred_fc_hip_cons
		tx_cred_fc_infinite : out std_logic_vector(5 downto 0);                      --               .tx_cred_fc_infinite
		tx_cred_hdr_fc      : out std_logic_vector(7 downto 0);                      --               .tx_cred_hdr_fc
		tx_cred_fc_sel      : in  std_logic_vector(1 downto 0)   := (others => '0'); --               .tx_cred_fc_sel
		tx_st_sop           : in  std_logic_vector(0 downto 0)   := (others => '0'); --          tx_st.startofpacket
		tx_st_eop           : in  std_logic_vector(0 downto 0)   := (others => '0'); --               .endofpacket
		tx_st_err           : in  std_logic_vector(0 downto 0)   := (others => '0'); --               .error
		tx_st_valid         : in  std_logic_vector(0 downto 0)   := (others => '0'); --               .valid
		tx_st_ready         : out std_logic;                                         --               .ready
		tx_st_data          : in  std_logic_vector(255 downto 0) := (others => '0'); --               .data
		tx_st_empty         : in  std_logic_vector(1 downto 0)   := (others => '0')  --               .empty
	);
end component;


component pcie_cfgbus 
    port(
		reset_n			: in std_logic;
		pld_clk			: in std_logic;
		tl_cfg_add		: in std_logic_vector(3 downto 0);
		tl_cfg_ctl		: in std_logic_vector(31 downto 0);
		
		cfg_busdev		: out	std_logic_vector(12 downto 0);
		cfg_dev_ctrl	: out std_logic_vector(31 downto 0);
		cfg_slot_ctrl	: out std_logic_vector(15 downto 0);
		cfg_link_ctrl	: out std_logic_vector(31 downto 0);
		cfg_prm_cmd		: out std_logic_vector(15 downto 0);	
		cfg_msi_addr	: out std_logic_vector(63 downto 0);
		cfg_pmcsr		: out std_logic_vector(31 downto 0);
		cfg_msixcsr		: out std_logic_vector(15 downto 0);
		cfg_msicsr		: out std_logic_vector(15 downto 0);
		tx_ercgen		: out	std_logic;
		rx_errcheck		: out std_logic;
		cfg_tcvcmap		: out std_logic_vector(23 downto 0);
		cfg_msi_data	: out std_logic_vector(15 downto 0)
    );
end component;

	component pcie_wram_narrow
        PORT
        (
                address_a               : IN STD_LOGIC_VECTOR (15 DOWNTO 0);
                address_b               : IN STD_LOGIC_VECTOR (15 DOWNTO 0);
                clock_a         : IN STD_LOGIC  := '1';
                clock_b         : IN STD_LOGIC ;
                data_a          : IN STD_LOGIC_VECTOR (31 DOWNTO 0);
                data_b          : IN STD_LOGIC_VECTOR (31 DOWNTO 0);
                wren_a          : IN STD_LOGIC  := '0';
                wren_b          : IN STD_LOGIC  := '0';
                q_a             : OUT STD_LOGIC_VECTOR (31 DOWNTO 0);
                q_b             : OUT STD_LOGIC_VECTOR (31 DOWNTO 0)
        );
	end component;

	component pcie_ram_narrow
	PORT
	(
		data					: IN STD_LOGIC_VECTOR (31 DOWNTO 0);
		rdaddress			: IN STD_LOGIC_VECTOR (13 DOWNTO 0);
		rdclock				: IN STD_LOGIC ;
		wraddress			: IN STD_LOGIC_VECTOR (15 DOWNTO 0);
		wrclock				: IN STD_LOGIC  := '1';
		wren					: IN STD_LOGIC  := '0';
		q						: OUT STD_LOGIC_VECTOR (127 DOWNTO 0)
	);
	end component;
	
	  component completer_fifo
	PORT
	(
		aclr		: IN STD_LOGIC ;
		clock		: IN STD_LOGIC ;
		data		: IN STD_LOGIC_VECTOR (47 DOWNTO 0);
		rdreq		: IN STD_LOGIC ;
		wrreq		: IN STD_LOGIC ;
		empty		: OUT STD_LOGIC ;
		full		: OUT STD_LOGIC ;
		q		: OUT STD_LOGIC_VECTOR (47 DOWNTO 0);
		usedw		: OUT STD_LOGIC_VECTOR (4 DOWNTO 0)
	);
end component;

component completer_wide_fifo
	PORT
	(
		aclr		: IN STD_LOGIC ;
		clock		: IN STD_LOGIC ;
		data		: IN STD_LOGIC_VECTOR (63 DOWNTO 0);
		rdreq		: IN STD_LOGIC ;
		wrreq		: IN STD_LOGIC ;
		empty		: OUT STD_LOGIC ;
		full		: OUT STD_LOGIC ;
		q		: OUT STD_LOGIC_VECTOR (63 DOWNTO 0);
		usedw		: OUT STD_LOGIC_VECTOR (4 DOWNTO 0)
	);
end component;



component dma_engine 
	generic (
			MEMWRITEADDRSIZE : integer := 14;
			MEMREADADDRSIZE  : integer := 12;
			MEMWRITEWIDTH	  : integer := 32;
			IRQNUM			 : std_logic_vector(4 downto 0) := "00000";
			ENABLE_BIT		 : integer := 0;
			NOW_BIT			 : integer := 0;
			ENABLE_INTERRUPT_BIT : integer := 0
		);
	port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
		
		-- Stuff for DMA writing
		dataclk:					in		std_logic;
		datain:					in 	std_logic_vector(MEMWRITEWIDTH-1 downto 0);
		datawren:				in		std_logic;
		endofevent:				in 	std_logic;
		memhalffull:			out	std_logic;
		
		-- Bus and device number
		cfg_busdev:				in		std_logic_vector(12 downto 0);
		
		-- Comunication with completer
		dma_request:			out	std_logic;
		dma_granted:			in		std_logic;
		dma_done:				out	std_logic;
		tx_ready:				in		std_logic;
		tx_data:					out	std_logic_vector(255 downto 0);
		tx_valid:				out	std_logic;
		tx_sop:					out	std_logic;
		tx_eop:					out	std_logic;
		tx_empty:				out	std_logic_vector(1 downto 0);
		
		-- Interrupt stuff
		app_msi_req:			out	std_logic;
		app_msi_tc:				out 	std_logic_vector(2 downto 0);
		app_msi_num:			out	std_logic_vector(4 downto 0);
		app_msi_ack:			in		std_logic;
		
		-- Configuration register
		dma_control_address:			in 	std_logic_vector(63 downto 0);
		dma_data_address: 			in 	std_logic_vector(63 downto 0);
		dma_data_address_out: 		out  	std_logic_vector(63 downto 0);
		dma_data_mem_addr:		 	in 	std_logic_vector(11 downto 0);
		dma_addrmem_data_written: 	in 	std_logic;
		dma_data_pages:				in 	std_logic_vector(19 downto 0);
		dma_data_pages_out:			out	std_logic_vector(19 downto 0);
		dma_data_n_addrs:				in 	std_logic_vector(11 downto 0);
		dma_register:					in    std_logic_vector(31 downto 0);
		dma_register_written:		in		std_logic;
		dma_status_register: 		out	std_logic_vector(31 downto 0);
		
		test_out:						out  	std_logic_vector(71 downto 0)
		);
end component;

component data_addrs_ram
			PORT	
			(
				address_a		: IN STD_LOGIC_VECTOR (11 DOWNTO 0);
				address_b		: IN STD_LOGIC_VECTOR (11 DOWNTO 0);
				clock				: IN STD_LOGIC  := '1';
				data_a			: IN STD_LOGIC_VECTOR (63 DOWNTO 0);
				data_b			: IN STD_LOGIC_VECTOR (63 DOWNTO 0);
				wren_a			: IN STD_LOGIC  := '0';
				wren_b			: IN STD_LOGIC  := '0';
				q_a				: OUT STD_LOGIC_VECTOR (63 DOWNTO 0);
				q_b				: OUT STD_LOGIC_VECTOR (63 DOWNTO 0)
			);
end component;	

component dma_ram
        PORT
        (
                data            : IN STD_LOGIC_VECTOR (255 DOWNTO 0);
                rdaddress       : IN STD_LOGIC_VECTOR (10 DOWNTO 0);
                rdclock         : IN STD_LOGIC ;
                wraddress       : IN STD_LOGIC_VECTOR (10 DOWNTO 0);
                wrclock         : IN STD_LOGIC  := '1';
                wren            : IN STD_LOGIC  := '0';
                q               : OUT STD_LOGIC_VECTOR (255 DOWNTO 0)
        );
end component;
component data_pages_ram
		PORT
		(
					address_a		: IN STD_LOGIC_VECTOR (11 DOWNTO 0);
					address_b		: IN STD_LOGIC_VECTOR (11 DOWNTO 0);
					clock		: IN STD_LOGIC  := '1';
					data_a		: IN STD_LOGIC_VECTOR (19 DOWNTO 0);
					data_b		: IN STD_LOGIC_VECTOR (19 DOWNTO 0);
					wren_a		: IN STD_LOGIC  := '0';
					wren_b		: IN STD_LOGIC  := '0';
					q_a		: OUT STD_LOGIC_VECTOR (19 DOWNTO 0);
					q_b		: OUT STD_LOGIC_VECTOR (19 DOWNTO 0)
		);
end component;

COMPONENT version_reg IS
    PORT (
        data_out: OUT STD_LOGIC_VECTOR(27 downto 0)
    );
END COMPONENT;

component dma_fifo
        PORT
        (
                aclr            : IN STD_LOGIC ;
                data            : IN STD_LOGIC_VECTOR (31 DOWNTO 0);
                rdclk           : IN STD_LOGIC ;
                rdreq           : IN STD_LOGIC ;
                wrclk           : IN STD_LOGIC ;
                wrreq           : IN STD_LOGIC ;
                q               : OUT STD_LOGIC_VECTOR (127 DOWNTO 0);
                rdempty         : OUT STD_LOGIC ;
                wrfull          : OUT STD_LOGIC 
        );
end component;




end package;
