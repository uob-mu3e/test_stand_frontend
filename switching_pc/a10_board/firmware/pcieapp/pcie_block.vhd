-----------------------------------------------------------------------------
-- Wrapper for the PCIe interface
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.pcie_components.all;

library altera;
use altera.altera_europa_support_lib.all;



entity pcie_block is 
generic (
    DMAMEMWRITEADDRSIZE : integer := 14;
    DMAMEMREADADDRSIZE  : integer := 12;
    DMAMEMWRITEWIDTH    : integer := 32;
    PCIE_X_g : positive := 8--;
);
port (
    o_writeregs_B               : out   reg32array;
	 o_regwritten_B:				out		std_logic_vector(63 downto 0);
    i_clk_B                     : in    std_logic := '0';
	 
	 o_writeregs_C               : out   reg32array;
	 o_regwritten_C:				out		std_logic_vector(63 downto 0);
    i_clk_C                     : in    std_logic := '0';

    local_rstn          : in    std_logic;
    appl_rstn           : in    std_logic;
    refclk              : in    std_logic;
    pcie_fastclk_out    : out   std_logic; -- 250 MHz clock

    -- PCI-Express (25 pins)
    pcie_rx_p           : in    std_logic_vector(PCIE_X_g-1 downto 0); --//PCIe Receive Data-req's OCT
    pcie_tx_p           : out   std_logic_vector(PCIE_X_g-1 downto 0); --//PCIe Transmit Data
    pcie_refclk_p       : in    std_logic; --//PCIe Clock- Terminate on MB
    pcie_led_g2         : out   std_logic; --//User LED - Labeled Gen2
    pcie_led_x1         : out   std_logic; --//User LED - Labeled x1
    pcie_led_x4         : out   std_logic; --//User LED - Labeled x4
    pcie_led_x8         : out   std_logic; --//User LED - Labeled x8
    pcie_perstn         : in    std_logic; --//PCIe Reset
    pcie_smbclk         : in    std_logic; --//SMBus Clock (TR=0)
    pcie_smbdat         : inout std_logic; --//SMBus Data (TR=0)
    pcie_waken          : out   std_logic; --//PCIe Wake-Up (TR=0)

    -- LEDs
    alive_led           : out   std_logic;
    comp_led            : out   std_logic;
    L0_led              : out   std_logic;

    -- pcie registers
    writeregs           : out   reg32array;
    regwritten          : out   std_logic_vector(63 downto 0);
    readregs            : in    reg32array;

    -- pcie writeable memory
    writememclk         : in    std_logic;
    writememreadaddr    : in    std_logic_vector(15 downto 0) := (others => '0');
    writememreaddata    : out   std_logic_vector(31 DOWNTO 0);

    -- pcie readable memory
    readmem_data        : in    std_logic_vector(31 downto 0) := (others => '0');
    readmem_addr        : in    std_logic_vector(15 downto 0) := (others => '0');
    readmemclk          : in    std_logic;
    readmem_wren        : in    std_logic := '0';
    readmem_endofevent  : in    std_logic := '0';

    -- dma memory
    dma_data            : in    std_logic_vector(DMAMEMWRITEWIDTH-1 downto 0) := (others => '0');
    dmamemclk           : in    std_logic;
    dmamem_wren         : in    std_logic := '0';
    dmamem_endofevent   : in    std_logic := '0';
    dmamemhalffull      : out   std_logic;

    -- second dma memory
    dma2_data           : in    std_logic_vector(DMAMEMWRITEWIDTH-1 downto 0) := (others => '0');
    dma2memclk          : in    std_logic;
    dma2mem_wren        : in    std_logic := '0';
    dma2mem_endofevent  : in    std_logic := '0';
    dma2memhalffull     : out   std_logic;

    -- test ports
    testout             : out   std_logic_vector(127 DOWNTO 0);
    testout_ena         : out   std_logic;
    pb_in               : in    std_logic_vector(2 downto 0) := (others => '0');
    inaddr32_r          : out   std_logic_vector(31 DOWNTO 0);
    inaddr32_w          : out   std_logic_vector(31 DOWNTO 0)
);
end entity;



architecture RTL of pcie_block is

	-- reset and clock stuff
	signal alive_cnt :  		STD_LOGIC_VECTOR (25 DOWNTO 0);
    signal any_rstn :  			STD_LOGIC;
	signal any_rstn_r :  		STD_LOGIC;
    signal any_rstn_rr :  		STD_LOGIC;
    signal pld_clk :			STD_LOGIC;
    signal rsnt_cntn : 			STD_LOGIC_VECTOR (10 DOWNTO 0);
	signal exits_r	:			STD_LOGIC;
    signal srst :				STD_LOGIC;
    signal app_rstn :			STD_LOGIC;
    signal srst0 :				STD_LOGIC;
    signal crst0 :				STD_LOGIC;
    signal app_rstn0 :			STD_LOGIC;
    
    -- Receiver IF
    signal rx_eqctrl_out :  	STD_LOGIC_VECTOR (3 DOWNTO 0);
    signal rx_eqdcgain_out : 	STD_LOGIC_VECTOR (2 DOWNTO 0);
    signal rx_mask0 :  			STD_LOGIC;
    signal rx_st_bardec0 :  	STD_LOGIC_VECTOR (7 DOWNTO 0);
    signal rx_st_be0 :  		STD_LOGIC_VECTOR (15 DOWNTO 0);
    signal rx_st_data0 :  		STD_LOGIC_VECTOR (255 DOWNTO 0);
    signal rx_st_empty0 :  		STD_LOGIC_vector(1 downto 0);
    signal rx_st_eop0 :  		STD_LOGIC_vector(0 downto 0);
    signal rx_st_sop0 :  		STD_LOGIC_vector(0 downto 0);
    signal rx_stream_data0 :  	STD_LOGIC_VECTOR (81 DOWNTO 0);
    signal rx_stream_data0_1 :  STD_LOGIC_VECTOR (81 DOWNTO 0);
    signal rx_st_ready0 :  	STD_LOGIC;
    signal rx_st_valid0 :  	STD_LOGIC_vector(0 downto 0);
    
    -- Transmitter IF
    
	signal tx_fifo_empty0 :  	STD_LOGIC_vector(0 downto 0);
    signal tx_preemp_0t_out :  	STD_LOGIC_VECTOR (4 DOWNTO 0);
    signal tx_preemp_1t_out :  	STD_LOGIC_VECTOR (4 DOWNTO 0);
    signal tx_preemp_2t_out :  	STD_LOGIC_VECTOR (4 DOWNTO 0);
    signal tx_st_data0 :  		STD_LOGIC_VECTOR (255 DOWNTO 0);
    signal tx_st_empty0 :  		STD_LOGIC_vector(1 downto 0);
    signal tx_st_eop0 :  		STD_LOGIC_vector(0 downto 0);
    signal tx_st_sop0 :  		STD_LOGIC_vector(0 downto 0);
    signal tx_stream_cred0 :  	STD_LOGIC_VECTOR (35 DOWNTO 0);
    signal tx_stream_data0 :  	STD_LOGIC_VECTOR (74 DOWNTO 0);
    signal tx_stream_data0_1 :  STD_LOGIC_VECTOR (74 DOWNTO 0);
    signal tx_st_ready0 :  	STD_LOGIC;
    signal tx_st_valid0 :  	STD_LOGIC_vector(0 downto 0);
    signal tx_vodctrl_out :  	STD_LOGIC_VECTOR (2 DOWNTO 0);
	 
	 signal tx_cred_datafccp   : std_logic_vector(11 downto 0);   
	 signal tx_cred_datafcnp   : std_logic_vector(11 downto 0); 
	 signal tx_cred_datafcp    : std_logic_vector(11 downto 0); 
	 signal tx_cred_fchipcons  : std_logic_vector(5 downto 0);  
	 signal tx_cred_fcinfinite : std_logic_vector(5 downto 0); 
	 signal tx_cred_hdrfccp    : std_logic_vector(7 downto 0);
	 signal tx_cred_hdrfcnp    : std_logic_vector(7 downto 0); 
	 signal tx_cred_hdrfcp     : std_logic_vector(7 downto 0);
	 
    
    -- LEDs
    signal lane_active_led : 	STD_LOGIC_VECTOR (3 DOWNTO 0);

	-- Test ports
	signal test_in :  			STD_LOGIC_VECTOR (31 DOWNTO 0);
    signal test_out_icm :  		STD_LOGIC_VECTOR (8 DOWNTO 0);

	-- Interrupt stuff
	signal app_int_ack :  	STD_LOGIC;
    signal app_int_sts :  	STD_LOGIC;
    signal app_msi_ack :  		STD_LOGIC;
    signal app_msi_num :  		STD_LOGIC_VECTOR (4 DOWNTO 0);
    signal app_msi_req :  		STD_LOGIC;
    signal app_msi_tc :  		STD_LOGIC_VECTOR (2 DOWNTO 0);
	signal pex_msi_num :  		STD_LOGIC_VECTOR (4 DOWNTO 0);
	
	-- Completion stuff
	signal cpl_err_icm : 		STD_LOGIC_VECTOR (6 DOWNTO 0);
    signal cpl_pending : 		STD_LOGIC;		

	-- LMI (Local Managment Interface)
	signal lmi_ack :  			STD_LOGIC;
	signal lmi_addr :  			STD_LOGIC_VECTOR (11 DOWNTO 0);
    signal lmi_din :  			STD_LOGIC_VECTOR (31 DOWNTO 0);
    signal lmi_dout :  			STD_LOGIC_VECTOR (31 DOWNTO 0);
    signal lmi_rden :  			STD_LOGIC;
    signal lmi_wren :  			STD_LOGIC;
    
    -- Power mamangement
    signal pme_to_sr : 			STD_LOGIC;
    
    -- Configuration space
	signal tl_cfg_add :  		STD_LOGIC_VECTOR (3 DOWNTO 0);
    signal tl_cfg_ctl :  		STD_LOGIC_VECTOR (31 DOWNTO 0);
    signal tl_cfg_ctl_wr :  	STD_LOGIC;
    signal tl_cfg_sts :  		STD_LOGIC_VECTOR (52 DOWNTO 0);
    signal tl_cfg_sts_wr :  	STD_LOGIC;
    
    -- Link training
	signal dl_ltssm :  			STD_LOGIC_VECTOR (4 DOWNTO 0);
    signal dl_ltssm_r :  		STD_LOGIC_VECTOR (4 DOWNTO 0);
   
    -- Config registers decoded
	signal cfg_busdev_icm :  	STD_LOGIC_VECTOR (12 DOWNTO 0);
    signal cfg_devcsr_icm :  	STD_LOGIC_VECTOR (31 DOWNTO 0);
    signal cfg_io_bas :  		STD_LOGIC_VECTOR (19 DOWNTO 0);
    signal cfg_linkcsr_icm :  	STD_LOGIC_VECTOR (31 DOWNTO 0);
    signal cfg_msicsr :  		STD_LOGIC_VECTOR (15 DOWNTO 0);
    signal cfg_np_bas :  		STD_LOGIC_VECTOR (11 DOWNTO 0);
    signal cfg_pr_bas :  		STD_LOGIC_VECTOR (43 DOWNTO 0);
    signal cfg_prmcsr_icm :  	STD_LOGIC_VECTOR (31 DOWNTO 0);

	-- Completion stuff after handling
	signal cpl_err_in : 		STD_LOGIC_VECTOR (6 DOWNTO 0);
	signal err_desc :  			STD_LOGIC_VECTOR (127 DOWNTO 0);
	
	-- Reconfig stuff after handling
	signal data_valid : 		STD_LOGIC;
	
	-- Reset and link status
	signal dlup : 			std_logic;
	signal dlup_exit: 	std_logic;
 	signal hotrst_exit: 	std_logic;
	signal l2_exit:		std_logic;
	signal currentspeed: std_logic_vector(1 downto 0);
	signal lane_act:		std_logic_vector(3 downto 0);
	signal serdes_pll_locked : std_logic;
	
	-- Application
	signal busy:				STD_LOGIC; 
	signal regloopback :reg32array;

	signal application_reset_n: std_logic;
	
	signal testbus :  			STD_LOGIC_VECTOR (127 DOWNTO 0);
	
begin

  any_rstn <= pcie_perstn and local_rstn;

  test_in(31 DOWNTO 9)  <= "00000000000000000000000";
  test_in(8 DOWNTO 5) 	<= "0101";
  test_in(4 DOWNTO 0) 	<= "01000";
  
  pcie_fastclk_out <= pld_clk;
  
  
  --reset Synchronizer
  process (pld_clk, any_rstn)
  begin
    if any_rstn = '0' then
      any_rstn_r <= std_logic'('1');
      any_rstn_rr <= std_logic'('1');
    elsif pld_clk'event and pld_clk = '1' then
      any_rstn_r <= std_logic'('0');
      any_rstn_rr <= any_rstn_r;
    end if;
  end process;
  
  --reset counter
  process (pld_clk, any_rstn_rr)
  begin
    if any_rstn_rr = '0' then
      rsnt_cntn <= "00000000000";
    elsif pld_clk'event and pld_clk = '1' then
      if (local_rstn = '0' OR dlup_exit = '0' OR hotrst_exit = '0' OR l2_exit = '0') then 
        rsnt_cntn <= "01111000000";
      elsif rsnt_cntn /= "10000000000" then 
        rsnt_cntn <= rsnt_cntn + '1';
      end if;
    end if;
  end process;

  srst <= srst0;
  
  --sync and config reset
  process (pld_clk, any_rstn_rr)
  begin
    if any_rstn_rr = '0' then
      app_rstn0 <= '0';
      srst0 <= '1';
      crst0 <= '1';
    elsif pld_clk'event and pld_clk = '1' then
      if (exits_r = '1') then 
        srst0 <= '1';
        crst0 <= '1';
        app_rstn0 <= '0';
      elsif rsnt_cntn = std_logic_vector'("10000000000") then 
        srst0 <= '0';
        crst0 <= '0';
        app_rstn0 <= '1';
      end if;
    end if;
  end process;

  --sync and config reset pipeline
  process (pld_clk, any_rstn_rr)
  begin
    if any_rstn_rr = '0' then
      app_rstn <= '0';
    elsif pld_clk'event and pld_clk = '1' then
      app_rstn <= app_rstn0;
    end if;
  end process;
  

  
  --LTSSM pipeline
  process (pld_clk, any_rstn_rr)
  begin
    if any_rstn_rr = '0' then
      dl_ltssm_r <= std_logic_vector'("00000");
    elsif pld_clk'event and pld_clk = '1' then
      dl_ltssm_r <= dl_ltssm;
    end if;

  end process;



  cpl_pending <= '0';
  
  pcie_led_x1 	<= lane_act(0);
  pcie_led_x4 	<= lane_act(2);
  pcie_led_x8 	<= lane_act(3);


    e_pcie : component work.cmp.pcie
	port map(
		clr_st              => open,  
		hpg_ctrler          => (others => '0'), -- only needed for root ports
		tl_cfg_add          => tl_cfg_add,
		tl_cfg_ctl          => tl_cfg_ctl,
		tl_cfg_sts          => tl_cfg_sts,
		cpl_err             => cpl_err_icm,
		cpl_pending         => cpl_pending,
		coreclkout_hip      => pld_clk,
		currentspeed        => currentspeed,
		pld_core_ready      => serdes_pll_locked,
		pld_clk_inuse       => open,  -- maybe this should be fed into the app reset
		serdes_pll_locked   => serdes_pll_locked,
		reset_status        => open,
		testin_zero         => open,
		rx_in0              => pcie_rx_p(0),
		rx_in1              => pcie_rx_p(1),
		rx_in2              => pcie_rx_p(2),
		rx_in3              => pcie_rx_p(3),
		rx_in4              => pcie_rx_p(4),
		rx_in5              => pcie_rx_p(5),
		rx_in6              => pcie_rx_p(6),
		rx_in7              => pcie_rx_p(7),
		tx_out0             => pcie_tx_p(0),
		tx_out1             => pcie_tx_p(1),
		tx_out2             => pcie_tx_p(2),
		tx_out3             => pcie_tx_p(3),
		tx_out4             => pcie_tx_p(4),
		tx_out5             => pcie_tx_p(5),
		tx_out6             => pcie_tx_p(6),
		tx_out7             => pcie_tx_p(7),
		derr_cor_ext_rcv    => open,
		derr_cor_ext_rpl    => open,
		derr_rpl            => open,
		dlup                => dlup,
		dlup_exit           => dlup_exit,
		ev128ns             => open,
		ev1us               => open,
		hotrst_exit         => hotrst_exit,
		int_status          => open,
		l2_exit             => l2_exit,
		lane_act            => lane_act,
		ltssmstate          => open,
		rx_par_err          => open,
		tx_par_err          => open,
		cfg_par_err         => open,
		ko_cpl_spc_header   => open,
		ko_cpl_spc_data     => open,
		app_int_sts         => app_int_sts,
		app_int_ack         => app_int_ack,
		app_msi_num         => app_msi_num,
		app_msi_req         => app_msi_req,
		app_msi_tc          => app_msi_tc,
		app_msi_ack         => app_msi_ack,
		npor                => pcie_perstn,
		pin_perst           => pcie_perstn,
		pld_clk             => pld_clk,
		pm_auxpwr           => '0',
		pm_data             => (others => '0'),
		pme_to_cr           => pme_to_sr,
		pm_event            => '0',
		pme_to_sr           => pme_to_sr,
		refclk              => refclk,
		rx_st_bar           => rx_st_bardec0,
		rx_st_mask          => rx_mask0,
		rx_st_sop           => rx_st_sop0,
		rx_st_eop           => rx_st_eop0,
		rx_st_err           => open,
		rx_st_valid         => rx_st_valid0,
		rx_st_ready         => rx_st_ready0,
		rx_st_data          => rx_st_data0,
		rx_st_empty         => rx_st_empty0,
		-- Credits, to be redone if needed
		tx_cred_data_fc     => open,
		tx_cred_fc_hip_cons => open,
		tx_cred_fc_infinite => open,
		tx_cred_hdr_fc      => open,
		tx_cred_fc_sel      => (others => '0'),
		tx_st_sop           => tx_st_sop0,
		tx_st_eop           => tx_st_eop0,
		tx_st_err           => open,
		tx_st_valid         => tx_st_valid0,
		tx_st_ready         => tx_st_ready0,
		tx_st_data          => tx_st_data0,
		tx_st_empty         => tx_st_empty0,
		
		-- Simulation only signals
		test_in             => X"00000188", --see age 102 in manual
		simu_mode_pipe      => '0', 
		sim_pipe_pclk_in    => '0',
		sim_pipe_rate       => open,
		sim_ltssmstate      => open,
		eidleinfersel0      => open,
		eidleinfersel1      => open,
		eidleinfersel2      => open,
		eidleinfersel3      => open,
		eidleinfersel4      => open,
		eidleinfersel5      => open,
		eidleinfersel6      => open,
		eidleinfersel7      => open,
		powerdown0          => open,
		powerdown1          => open,
		powerdown2          => open,
		powerdown3          => open,
		powerdown4          => open,
		powerdown5          => open,
		powerdown6          => open,
		powerdown7          => open,
		rxpolarity0         => open,
		rxpolarity1         => open,
		rxpolarity2         => open,
		rxpolarity3         => open,
		rxpolarity4         => open,
		rxpolarity5         => open,
		rxpolarity6         => open,
		rxpolarity7         => open,
		txcompl0            => open,
		txcompl1            => open,
		txcompl2            => open,
		txcompl3            => open,
		txcompl4            => open,
		txcompl5            => open,
		txcompl6            => open,
		txcompl7            => open,
		txdata0             => open,
		txdata1             => open,
		txdata2             => open,
		txdata3             => open,
		txdata4             => open,
		txdata5             => open,
		txdata6             => open,
		txdata7             => open,
		txdatak0            => open,
		txdatak1            => open,
		txdatak2            => open,
		txdatak3            => open,
		txdatak4            => open,
		txdatak5            => open,
		txdatak6            => open,
		txdatak7            => open,
		txdetectrx0         => open,
		txdetectrx1         => open,
		txdetectrx2         => open,
		txdetectrx3         => open,
		txdetectrx4         => open,
		txdetectrx5         => open,
		txdetectrx6         => open,
		txdetectrx7         => open,
		txelecidle0         => open,
		txelecidle1         => open,
		txelecidle2         => open,
		txelecidle3         => open,
		txelecidle4         => open,
		txelecidle5         => open,
		txelecidle6         => open,
		txelecidle7         => open,
		txdeemph0           => open,
		txdeemph1           => open,
		txdeemph2           => open,
		txdeemph3           => open,
		txdeemph4           => open,
		txdeemph5           => open,
		txdeemph6           => open,
		txdeemph7           => open,
		txmargin0           => open,
		txmargin1           => open,
		txmargin2           => open,
		txmargin3           => open,
		txmargin4           => open,
		txmargin5           => open,
		txmargin6           => open,
		txmargin7           => open,
		txswing0            => open,
		txswing1            => open,
		txswing2            => open,
		txswing3            => open,
		txswing4            => open,
		txswing5            => open,
		txswing6            => open,
		txswing7            => open,
		phystatus0          => '0',
		phystatus1          => '0',
		phystatus2          => '0',
		phystatus3          => '0',
		phystatus4          => '0',
		phystatus5          => '0',
		phystatus6          => '0',
		phystatus7          => '0',
		rxdata0             => (others => '0'),
		rxdata1             => (others => '0'),
		rxdata2             => (others => '0'),
		rxdata3             => (others => '0'),
		rxdata4             => (others => '0'),
		rxdata5             => (others => '0'),
		rxdata6             => (others => '0'),
		rxdata7             => (others => '0'),
		rxdatak0            => (others => '0'),
		rxdatak1            => (others => '0'),
		rxdatak2            => (others => '0'),
		rxdatak3            => (others => '0'),
		rxdatak4            => (others => '0'),
		rxdatak5            => (others => '0'),
		rxdatak6            => (others => '0'),
		rxdatak7            => (others => '0'),
		rxelecidle0         => '0',
		rxelecidle1         => '0',
		rxelecidle2         => '0',
		rxelecidle3         => '0',
		rxelecidle4         => '0',
		rxelecidle5         => '0',
		rxelecidle6         => '0',
		rxelecidle7         => '0',
		rxstatus0           => (others => '0'),
		rxstatus1           => (others => '0'),
		rxstatus2           => (others => '0'),
		rxstatus3           => (others => '0'),
		rxstatus4           => (others => '0'),
		rxstatus5           => (others => '0'),
		rxstatus6           => (others => '0'),
		rxstatus7           => (others => '0'),
		rxvalid0            => '0',
		rxvalid1            => '0',
		rxvalid2            => '0',
		rxvalid3            => '0',
		rxvalid4            => '0',
		rxvalid5            => '0',
		rxvalid6            => '0',
		rxvalid7            => '0',
		rxdataskip0         => '0',
		rxdataskip1         => '0',
		rxdataskip2         => '0',
		rxdataskip3         => '0',
		rxdataskip4         => '0',
		rxdataskip5         => '0',
		rxdataskip6         => '0',
		rxdataskip7         => '0',
		rxblkst0            => '0',
		rxblkst1            => '0',
		rxblkst2            => '0',
		rxblkst3            => '0',
		rxblkst4            => '0',
		rxblkst5            => '0',
		rxblkst6            => '0',
		rxblkst7            => '0',
		rxsynchd0           => (others => '0'),
		rxsynchd1           => (others => '0'),
		rxsynchd2           => (others => '0'),
		rxsynchd3           => (others => '0'),
		rxsynchd4           => (others => '0'),
		rxsynchd5           => (others => '0'),
		rxsynchd6           => (others => '0'),
		rxsynchd7           => (others => '0'),
		currentcoeff0       => open,
		currentcoeff1       => open,
		currentcoeff2       => open,
		currentcoeff3       => open,
		currentcoeff4       => open,
		currentcoeff5       => open,
		currentcoeff6       => open,
		currentcoeff7       => open,
		currentrxpreset0    => open,
		currentrxpreset1    => open,
		currentrxpreset2    => open,
		currentrxpreset3    => open,
		currentrxpreset4    => open,
		currentrxpreset5    => open,
		currentrxpreset6    => open,
		currentrxpreset7    => open,
		txsynchd0           => open,
		txsynchd1           => open,
		txsynchd2           => open,
		txsynchd3           => open,
		txsynchd4           => open,
		txsynchd5           => open,
		txsynchd6           => open,
		txsynchd7           => open,
		txblkst0            => open,
		txblkst1            => open,
		txblkst2            => open,
		txblkst3            => open,
		txblkst4            => open,
		txblkst5            => open,
		txblkst6            => open,
		txblkst7            => open,
		txdataskip0         => open,
		txdataskip1         => open,
		txdataskip2         => open,
		txdataskip3         => open,
		txdataskip4         => open,
		txdataskip5         => open,
		txdataskip6         => open,
		txdataskip7         => open,
		rate0               => open,
		rate1               => open,
		rate2               => open,
		rate3               => open,
		rate4               => open,
		rate5               => open,
		rate6               => open,
		rate7               => open			
	);


  

  


-- Configuration bus decode
    cfgbus: work.pcie_cfgbus
    port map(
		reset_n			=> pcie_perstn,
		pld_clk			=> pld_clk,
		tl_cfg_add		=> tl_cfg_add,
		tl_cfg_ctl		=> tl_cfg_ctl,
		
		cfg_busdev		=> cfg_busdev_icm,
		cfg_dev_ctrl	=> open,
		cfg_slot_ctrl	=> open,
		cfg_link_ctrl	=> open,
		cfg_prm_cmd		=> open,
		cfg_msi_addr	=> open,
		cfg_pmcsr		=> open,
		cfg_msixcsr		=> open,
		cfg_msicsr		=> open,
		tx_ercgen		=> open,
		rx_errcheck		=> open,
		cfg_tcvcmap		=> open,
		cfg_msi_data	=> open
    );


	 application_reset_n <= '0' when local_rstn = '0' or appl_rstn = '0' else '1';
	 
    e_pcie_application : entity work.pcie_application
    generic map(
			DMAMEMWRITEADDRSIZE => DMAMEMWRITEADDRSIZE,
			DMAMEMREADADDRSIZE  => DMAMEMREADADDRSIZE,
			DMAMEMWRITEWIDTH	  => DMAMEMWRITEWIDTH
    )
    port map (
        o_writeregs_B           => o_writeregs_B,
		  o_regwritten_B			  => o_regwritten_B,
        i_clk_B                 => i_clk_B,
		  
		  o_writeregs_C           => o_writeregs_C,
		  o_regwritten_C			  => o_regwritten_C,
        i_clk_C                 => i_clk_C,

		local_rstn			=> application_reset_n,
		refclk				=> pld_clk,
	
		-- to IF
		tx_st_data0 		=> tx_st_data0,
		tx_st_eop0 			=> tx_st_eop0(0),
		tx_st_sop0 			=> tx_st_sop0(0),
		tx_st_ready0 		=> tx_st_ready0,
		tx_st_valid0 		=> tx_st_valid0(0),
		tx_st_empty0 		=> tx_st_empty0,
		
		-- from Config
		completer_id 		=> cfg_busdev_icm,
		
		-- from IF
		rx_st_data0 		=> rx_st_data0,
		rx_st_eop0 			=> rx_st_eop0(0),
		rx_st_sop0 			=>	rx_st_sop0(0),
		rx_st_ready0 		=> rx_st_ready0,
		rx_st_valid0 		=> rx_st_valid0(0),
		rx_bar0 				=> rx_st_bardec0,
		
		-- Interrupt stuff
		app_msi_req			=> app_msi_req,
		app_msi_tc			=> app_msi_tc,
		app_msi_num			=> app_msi_num,
		app_msi_ack			=> app_msi_ack,
		
		-- registers
		writeregs 			=> writeregs,
		regwritten			=> regwritten,
		readregs 			=>	readregs,
		
			-- pcie writeable memory
		writememclk		  	=> writememclk,
		writememreadaddr 	=> writememreadaddr,
		writememreaddata 	=> writememreaddata,
		
				-- pcie readable memory
		readmem_data 		=> readmem_data,
		readmem_addr 		=> readmem_addr,
		readmemclk			=> readmemclk,
		readmem_wren		=> readmem_wren,
		readmem_endofevent => readmem_endofevent,
		
						-- dma memory
		dma_data 			=> dma_data,
		dmamemclk			=> dmamemclk,
		dmamem_wren			=> dmamem_wren,
		dmamem_endofevent	=> dmamem_endofevent,
		dmamemhalffull		=> dmamemhalffull,
		
					-- 2nd dma memory
		dma2_data 			=> dma2_data,
		dma2memclk			=> dma2memclk,
		dma2mem_wren			=> dma2mem_wren,
		dma2mem_endofevent	=> dma2mem_endofevent,
		dma2memhalffull		=> dma2memhalffull,
		
				-- test ports  
		testout				=> testout,
		testin				=> testbus,
		testout_ena			=> testout_ena,
		pb_in					=> pb_in,
		inaddr32_r			=> inaddr32_r,
		inaddr32_w			=> inaddr32_w
	);
	



end architecture;
