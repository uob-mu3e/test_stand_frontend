

library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use work.ipbus.ALL;
use ieee.numeric_std.all;
Library UNISIM;
use UNISIM.vcomponents.all;

entity mu3e_top is 
	port(
        sysclk_p, sysclk_n: in STD_LOGIC;
        leds: out STD_LOGIC_VECTOR(7 downto 0);
        phy_rstb : out STD_LOGIC;
        TX_MGT_RST_L: out STD_LOGIC;
        TX_MGT_SLCT_L: out STD_LOGIC;
        TX_CLK_RST_L: out STD_LOGIC;
        TX_CLK_SLCT_L: out STD_LOGIC;
        CLK_OE_L: out STD_LOGIC;
        CLK_RST_L: out STD_LOGIC;
--        rstb: in STD_LOGIC;
--        dip_switch: in std_logic_vector(3 downto 0);
        Q1_CLK1_GTREFCLK_PAD_N_IN               : in   std_logic;
   		Q1_CLK1_GTREFCLK_PAD_P_IN               : in   std_logic;
        Q1_CLK0_GTREFCLK_PAD_N_IN               : in   std_logic;
   		Q1_CLK0_GTREFCLK_PAD_P_IN               : in   std_logic;
--        SCL: inout std_logic;
--        SDA: inout std_logic;
--		RXP_IN: in std_logic_vector(7 downto 0);
--		RXN_IN: in std_logic_vector(7 downto 0);
		TXP_OUT: out std_logic_vector(7 downto 0);
		TXN_OUT: out std_logic_vector(7 downto 0);
		rgmii_txd: out std_logic_vector(3 downto 0);
		rgmii_tx_ctl: out std_logic;
		rgmii_txc: out std_logic;
		rgmii_rxd: in std_logic_vector(3 downto 0);
		rgmii_rx_ctl: in std_logic;
		rgmii_rxc: in std_logic
        );
end mu3e_top;

architecture rtl of mu3e_top is

        signal clk125, clk200,clk125_90, ipb_clk, locked, rst_125, rst_ipb, onehz, nuke,soft_rst,rst125,rst_ipb_ctrl : STD_LOGIC;
        signal mac_tx_data, mac_rx_data: std_logic_vector(7 downto 0);
        signal mac_tx_valid, mac_tx_last, mac_tx_ready, mac_rx_valid, mac_rx_last, mac_rx_error: std_logic;
        signal ipb_master_out : ipb_wbus;
        signal ipb_master_in : ipb_rbus;
        signal mac_addr: std_logic_vector(47 downto 0);
        signal ip_addr: std_logic_vector(31 downto 0);
        signal mac_tx_error : std_logic;

        signal led_ipbus : std_logic_vector(1 downto 0);

        

component gtx_reset_firefly is
port
(
    SOFT_RESET_TX_IN                        : in   std_logic;
    DONT_RESET_ON_DATA_ERROR_IN             : in   std_logic;
    Q1_CLK0_GTREFCLK_PAD_N_IN               : in   std_logic;
    Q1_CLK0_GTREFCLK_PAD_P_IN               : in   std_logic;
    --Q1_CLK1_GTREFCLK_PAD_N_IN               : in   std_logic;
    --Q1_CLK1_GTREFCLK_PAD_P_IN               : in   std_logic;

    GT0_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT0_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT0_DATA_VALID_IN                       : in   std_logic;
    GT1_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT1_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT1_DATA_VALID_IN                       : in   std_logic;
    GT2_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT2_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT2_DATA_VALID_IN                       : in   std_logic;
    GT3_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT3_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT3_DATA_VALID_IN                       : in   std_logic;
    GT4_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT4_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT4_DATA_VALID_IN                       : in   std_logic;
    GT5_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT5_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT5_DATA_VALID_IN                       : in   std_logic;
    GT6_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT6_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT6_DATA_VALID_IN                       : in   std_logic;
    GT7_TX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT7_RX_FSM_RESET_DONE_OUT               : out  std_logic;
    GT7_DATA_VALID_IN                       : in   std_logic;
 
    GT0_TXUSRCLK_OUT                        : out  std_logic;
    GT0_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT1_TXUSRCLK_OUT                        : out  std_logic;
    GT1_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT2_TXUSRCLK_OUT                        : out  std_logic;
    GT2_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT3_TXUSRCLK_OUT                        : out  std_logic;
    GT3_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT4_TXUSRCLK_OUT                        : out  std_logic;
    GT4_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT5_TXUSRCLK_OUT                        : out  std_logic;
    GT5_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT6_TXUSRCLK_OUT                        : out  std_logic;
    GT6_TXUSRCLK2_OUT                       : out  std_logic;
 
    GT7_TXUSRCLK_OUT                        : out  std_logic;
    GT7_TXUSRCLK2_OUT                       : out  std_logic;

    --_________________________________________________________________________
    --GT0  (X1Y2)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt0_cpllfbclklost_out                   : out  std_logic;
    gt0_cplllock_out                        : out  std_logic;
    gt0_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt0_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt0_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt0_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt0_drpen_in                            : in   std_logic;
    gt0_drprdy_out                          : out  std_logic;
    gt0_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt0_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt0_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt0_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt0_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt0_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt0_eyescandataerror_out                : out  std_logic;
    gt0_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt0_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt0_rxprbserr_out                       : out  std_logic;
    gt0_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt0_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt0_rxbufreset_in                       : in   std_logic;
    gt0_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt0_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt0_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt0_gtrxreset_in                        : in   std_logic;
    gt0_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt0_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt0_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt0_gttxreset_in                        : in   std_logic;
    gt0_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt0_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt0_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt0_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt0_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt0_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt0_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt0_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt0_gtxtxn_out                          : out  std_logic;
    gt0_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt0_txoutclkfabric_out                  : out  std_logic;
    gt0_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt0_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt0_txpcsreset_in                       : in   std_logic;
    gt0_txpmareset_in                       : in   std_logic;
    gt0_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt0_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt0_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT1  (X1Y3)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt1_cpllfbclklost_out                   : out  std_logic;
    gt1_cplllock_out                        : out  std_logic;
    gt1_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt1_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt1_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt1_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt1_drpen_in                            : in   std_logic;
    gt1_drprdy_out                          : out  std_logic;
    gt1_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt1_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt1_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt1_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt1_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt1_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt1_eyescandataerror_out                : out  std_logic;
    gt1_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt1_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt1_rxprbserr_out                       : out  std_logic;
    gt1_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt1_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt1_rxbufreset_in                       : in   std_logic;
    gt1_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt1_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt1_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt1_gtrxreset_in                        : in   std_logic;
    gt1_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt1_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt1_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt1_gttxreset_in                        : in   std_logic;
    gt1_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt1_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt1_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt1_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt1_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt1_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt1_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt1_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt1_gtxtxn_out                          : out  std_logic;
    gt1_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt1_txoutclkfabric_out                  : out  std_logic;
    gt1_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt1_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt1_txpcsreset_in                       : in   std_logic;
    gt1_txpmareset_in                       : in   std_logic;
    gt1_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt1_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt1_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT2  (X1Y4)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt2_cpllfbclklost_out                   : out  std_logic;
    gt2_cplllock_out                        : out  std_logic;
    gt2_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt2_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt2_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt2_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt2_drpen_in                            : in   std_logic;
    gt2_drprdy_out                          : out  std_logic;
    gt2_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt2_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt2_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt2_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt2_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt2_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt2_eyescandataerror_out                : out  std_logic;
    gt2_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt2_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt2_rxprbserr_out                       : out  std_logic;
    gt2_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt2_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt2_rxbufreset_in                       : in   std_logic;
    gt2_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt2_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt2_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt2_gtrxreset_in                        : in   std_logic;
    gt2_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt2_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt2_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt2_gttxreset_in                        : in   std_logic;
    gt2_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt2_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt2_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt2_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt2_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt2_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt2_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt2_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt2_gtxtxn_out                          : out  std_logic;
    gt2_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt2_txoutclkfabric_out                  : out  std_logic;
    gt2_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt2_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt2_txpcsreset_in                       : in   std_logic;
    gt2_txpmareset_in                       : in   std_logic;
    gt2_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt2_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt2_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT3  (X1Y5)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt3_cpllfbclklost_out                   : out  std_logic;
    gt3_cplllock_out                        : out  std_logic;
    gt3_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt3_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt3_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt3_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt3_drpen_in                            : in   std_logic;
    gt3_drprdy_out                          : out  std_logic;
    gt3_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt3_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt3_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt3_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt3_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt3_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt3_eyescandataerror_out                : out  std_logic;
    gt3_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt3_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt3_rxprbserr_out                       : out  std_logic;
    gt3_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt3_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt3_rxbufreset_in                       : in   std_logic;
    gt3_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt3_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt3_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt3_gtrxreset_in                        : in   std_logic;
    gt3_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt3_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt3_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt3_gttxreset_in                        : in   std_logic;
    gt3_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt3_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt3_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt3_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt3_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt3_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt3_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt3_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt3_gtxtxn_out                          : out  std_logic;
    gt3_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt3_txoutclkfabric_out                  : out  std_logic;
    gt3_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt3_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt3_txpcsreset_in                       : in   std_logic;
    gt3_txpmareset_in                       : in   std_logic;
    gt3_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt3_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt3_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT4  (X1Y6)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt4_cpllfbclklost_out                   : out  std_logic;
    gt4_cplllock_out                        : out  std_logic;
    gt4_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt4_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt4_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt4_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt4_drpen_in                            : in   std_logic;
    gt4_drprdy_out                          : out  std_logic;
    gt4_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt4_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt4_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt4_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt4_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt4_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt4_eyescandataerror_out                : out  std_logic;
    gt4_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt4_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt4_rxprbserr_out                       : out  std_logic;
    gt4_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt4_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt4_rxbufreset_in                       : in   std_logic;
    gt4_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt4_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt4_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt4_gtrxreset_in                        : in   std_logic;
    gt4_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt4_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt4_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt4_gttxreset_in                        : in   std_logic;
    gt4_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt4_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt4_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt4_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt4_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt4_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt4_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt4_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt4_gtxtxn_out                          : out  std_logic;
    gt4_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt4_txoutclkfabric_out                  : out  std_logic;
    gt4_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt4_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt4_txpcsreset_in                       : in   std_logic;
    gt4_txpmareset_in                       : in   std_logic;
    gt4_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt4_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt4_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT5  (X1Y7)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt5_cpllfbclklost_out                   : out  std_logic;
    gt5_cplllock_out                        : out  std_logic;
    gt5_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt5_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt5_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt5_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt5_drpen_in                            : in   std_logic;
    gt5_drprdy_out                          : out  std_logic;
    gt5_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt5_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt5_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt5_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt5_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt5_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt5_eyescandataerror_out                : out  std_logic;
    gt5_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt5_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt5_rxprbserr_out                       : out  std_logic;
    gt5_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt5_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt5_rxbufreset_in                       : in   std_logic;
    gt5_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt5_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt5_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt5_gtrxreset_in                        : in   std_logic;
    gt5_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt5_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt5_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt5_gttxreset_in                        : in   std_logic;
    gt5_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt5_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt5_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt5_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt5_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt5_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt5_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt5_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt5_gtxtxn_out                          : out  std_logic;
    gt5_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt5_txoutclkfabric_out                  : out  std_logic;
    gt5_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt5_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt5_txpcsreset_in                       : in   std_logic;
    gt5_txpmareset_in                       : in   std_logic;
    gt5_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt5_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt5_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT6  (X1Y8)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt6_cpllfbclklost_out                   : out  std_logic;
    gt6_cplllock_out                        : out  std_logic;
    gt6_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt6_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt6_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt6_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt6_drpen_in                            : in   std_logic;
    gt6_drprdy_out                          : out  std_logic;
    gt6_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt6_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt6_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt6_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt6_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt6_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt6_eyescandataerror_out                : out  std_logic;
    gt6_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt6_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt6_rxprbserr_out                       : out  std_logic;
    gt6_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt6_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt6_rxbufreset_in                       : in   std_logic;
    gt6_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt6_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt6_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt6_gtrxreset_in                        : in   std_logic;
    gt6_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt6_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt6_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt6_gttxreset_in                        : in   std_logic;
    gt6_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt6_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt6_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt6_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt6_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt6_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt6_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt6_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt6_gtxtxn_out                          : out  std_logic;
    gt6_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt6_txoutclkfabric_out                  : out  std_logic;
    gt6_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt6_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt6_txpcsreset_in                       : in   std_logic;
    gt6_txpmareset_in                       : in   std_logic;
    gt6_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt6_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt6_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --GT7  (X1Y9)
    --____________________________CHANNEL PORTS________________________________
    --------------------------------- CPLL Ports -------------------------------
    gt7_cpllfbclklost_out                   : out  std_logic;
    gt7_cplllock_out                        : out  std_logic;
    gt7_cpllreset_in                        : in   std_logic;
    ---------------------------- Channel - DRP Ports  --------------------------
    gt7_drpaddr_in                          : in   std_logic_vector(8 downto 0);
    gt7_drpdi_in                            : in   std_logic_vector(15 downto 0);
    gt7_drpdo_out                           : out  std_logic_vector(15 downto 0);
    gt7_drpen_in                            : in   std_logic;
    gt7_drprdy_out                          : out  std_logic;
    gt7_drpwe_in                            : in   std_logic;
    --------------------------- Digital Monitor Ports --------------------------
    gt7_dmonitorout_out                     : out  std_logic_vector(7 downto 0);
    ------------------------------- Loopback Ports -----------------------------
    gt7_loopback_in                         : in   std_logic_vector(2 downto 0);
    ------------------------------ Power-Down Ports ----------------------------
    gt7_rxpd_in                             : in   std_logic_vector(1 downto 0);
    gt7_txpd_in                             : in   std_logic_vector(1 downto 0);
    --------------------- RX Initialization and Reset Ports --------------------
    gt7_eyescanreset_in                     : in   std_logic;
    -------------------------- RX Margin Analysis Ports ------------------------
    gt7_eyescandataerror_out                : out  std_logic;
    gt7_eyescantrigger_in                   : in   std_logic;
    ------------------------- Receive Ports - CDR Ports ------------------------
    gt7_rxcdrovrden_in                      : in   std_logic;
    ------------------- Receive Ports - Pattern Checker Ports ------------------
    gt7_rxprbserr_out                       : out  std_logic;
    gt7_rxprbssel_in                        : in   std_logic_vector(2 downto 0);
    ------------------- Receive Ports - Pattern Checker ports ------------------
    gt7_rxprbscntreset_in                   : in   std_logic;
    ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
    gt7_rxbufreset_in                       : in   std_logic;
    gt7_rxbufstatus_out                     : out  std_logic_vector(2 downto 0);
    --------------------- Receive Ports - RX Equalizer Ports -------------------
    gt7_rxmonitorout_out                    : out  std_logic_vector(6 downto 0);
    gt7_rxmonitorsel_in                     : in   std_logic_vector(1 downto 0);
    ------------- Receive Ports - RX Initialization and Reset Ports ------------
    gt7_gtrxreset_in                        : in   std_logic;
    gt7_rxpcsreset_in                       : in   std_logic;
    ------------------------ TX Configurable Driver Ports ----------------------
    gt7_txpostcursor_in                     : in   std_logic_vector(4 downto 0);
    gt7_txprecursor_in                      : in   std_logic_vector(4 downto 0);
    --------------------- TX Initialization and Reset Ports --------------------
    gt7_gttxreset_in                        : in   std_logic;
    gt7_txuserrdy_in                        : in   std_logic;
    ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
    gt7_txchardispmode_in                   : in   std_logic_vector(3 downto 0);
    gt7_txchardispval_in                    : in   std_logic_vector(3 downto 0);
    ------------------ Transmit Ports - Pattern Generator Ports ----------------
    gt7_txprbsforceerr_in                   : in   std_logic;
    ---------------------- Transmit Ports - TX Buffer Ports --------------------
    gt7_txbufstatus_out                     : out  std_logic_vector(1 downto 0);
    --------------- Transmit Ports - TX Configurable Driver Ports --------------
    gt7_txdiffctrl_in                       : in   std_logic_vector(3 downto 0);
    gt7_txmaincursor_in                     : in   std_logic_vector(6 downto 0);
    ------------------ Transmit Ports - TX Data Path interface -----------------
    gt7_txdata_in                           : in   std_logic_vector(31 downto 0);
    ---------------- Transmit Ports - TX Driver and OOB signaling --------------
    gt7_gtxtxn_out                          : out  std_logic;
    gt7_gtxtxp_out                          : out  std_logic;
    ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
    gt7_txoutclkfabric_out                  : out  std_logic;
    gt7_txoutclkpcs_out                     : out  std_logic;
    --------------------- Transmit Ports - TX Gearbox Ports --------------------
    gt7_txcharisk_in                        : in   std_logic_vector(3 downto 0);
    ------------- Transmit Ports - TX Initialization and Reset Ports -----------
    gt7_txpcsreset_in                       : in   std_logic;
    gt7_txpmareset_in                       : in   std_logic;
    gt7_txresetdone_out                     : out  std_logic;
    ----------------- Transmit Ports - TX Polarity Control Ports ---------------
    gt7_txpolarity_in                       : in   std_logic;
    ------------------ Transmit Ports - pattern Generator Ports ----------------
    gt7_txprbssel_in                        : in   std_logic_vector(2 downto 0);

    --____________________________COMMON PORTS________________________________
     GT0_QPLLOUTCLK_OUT  : out std_logic;
     GT0_QPLLOUTREFCLK_OUT : out std_logic;
    --____________________________COMMON PORTS________________________________
     GT1_QPLLOUTCLK_OUT  : out std_logic;
     GT1_QPLLOUTREFCLK_OUT : out std_logic;
    --____________________________COMMON PORTS________________________________
     GT2_QPLLOUTCLK_OUT  : out std_logic;
     GT2_QPLLOUTREFCLK_OUT : out std_logic;

          sysclk_in                               : in   std_logic

);

end component;

signal tied_to_ground_i : std_logic;
signal tied_to_vcc_i : std_logic;


signal gt0_txfsmresetdone_i : std_logic;
signal gt0_rxfsmresetdone_i : std_logic;
signal gt0_track_data_i : std_logic;
signal gt0_txusrclk_i : std_logic;
signal gt0_txusrclk2_i : std_logic;
signal gt0_rxusrclk_i : std_logic;
signal gt0_rxusrclk2_i : std_logic;
signal gt0_cpllfbclklost_i : std_logic;
signal gt0_cplllock_i : std_logic;
signal gt0_drpaddr_i : std_logic_vector(8 downto 0);
signal gt0_drpdi_i : std_logic_vector(15 downto 0);
signal gt0_drpdo_i : std_logic_vector(15 downto 0);
signal gt0_drpen_i : std_logic;
signal gt0_drprdy_i : std_logic;
signal gt0_drpwe_i : std_logic;
signal gt0_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt0_rxpd_i : std_logic_vector(1 downto 0);
signal gt0_txpd_i : std_logic_vector(1 downto 0);
signal gt0_eyescandataerror_i : std_logic;
signal gt0_rxcdrhold_i : std_logic;
signal gt0_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt0_rxdata_i : std_logic_vector(31 downto 0);
signal gt0_rxprbserr_i : std_logic;
signal gt0_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt0_rxprbscntreset_i : std_logic;
signal gt0_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt0_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt0_rxbufreset_i : std_logic;
signal gt0_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt0_rxbyteisaligned_i : std_logic;
signal gt0_rxbyterealign_i : std_logic;
signal gt0_rxcommadet_i : std_logic;
signal gt0_rxmcommaalignen_i : std_logic;
signal gt0_rxpcommaalignen_i : std_logic;
signal gt0_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt0_rxoutclkfabric_i : std_logic;
signal gt0_rxpmareset_i : std_logic;
signal gt0_rxlpmen_i : std_logic;
signal gt0_rxpolarity_i : std_logic;
signal gt0_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt0_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt0_rxresetdone_i : std_logic;
signal gt0_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt0_txprecursor_i : std_logic_vector(4 downto 0);
signal gt0_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt0_txchardispval_i : std_logic_vector(3 downto 0);
signal gt0_txprbsforceerr_i : std_logic;
signal gt0_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt0_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt0_txdata_i : std_logic_vector(31 downto 0);
signal gt0_txoutclkfabric_i : std_logic;
signal gt0_txoutclkpcs_i : std_logic;
signal gt0_txcharisk_i : std_logic_vector(3 downto 0);
signal gt0_txresetdone_i : std_logic;
signal gt0_txpolarity_i : std_logic;
signal gt0_txprbssel_i : std_logic_vector(2 downto 0);

signal gt1_txfsmresetdone_i : std_logic;
signal gt1_rxfsmresetdone_i : std_logic;
signal gt1_track_data_i : std_logic;
signal gt1_txusrclk_i : std_logic;
signal gt1_txusrclk2_i : std_logic;
signal gt1_rxusrclk_i : std_logic;
signal gt1_rxusrclk2_i : std_logic;
signal gt1_cpllfbclklost_i : std_logic;
signal gt1_cplllock_i : std_logic;
signal gt1_drpaddr_i : std_logic_vector(8 downto 0);
signal gt1_drpdi_i : std_logic_vector(15 downto 0);
signal gt1_drpdo_i : std_logic_vector(15 downto 0);
signal gt1_drpen_i : std_logic;
signal gt1_drprdy_i : std_logic;
signal gt1_drpwe_i : std_logic;
signal gt1_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt1_rxpd_i : std_logic_vector(1 downto 0);
signal gt1_txpd_i : std_logic_vector(1 downto 0);
signal gt1_eyescandataerror_i : std_logic;
signal gt1_rxcdrhold_i : std_logic;
signal gt1_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt1_rxdata_i : std_logic_vector(31 downto 0);
signal gt1_rxprbserr_i : std_logic;
signal gt1_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt1_rxprbscntreset_i : std_logic;
signal gt1_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt1_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt1_rxbufreset_i : std_logic;
signal gt1_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt1_rxbyteisaligned_i : std_logic;
signal gt1_rxbyterealign_i : std_logic;
signal gt1_rxcommadet_i : std_logic;
signal gt1_rxmcommaalignen_i : std_logic;
signal gt1_rxpcommaalignen_i : std_logic;
signal gt1_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt1_rxoutclkfabric_i : std_logic;
signal gt1_rxpmareset_i : std_logic;
signal gt1_rxlpmen_i : std_logic;
signal gt1_rxpolarity_i : std_logic;
signal gt1_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt1_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt1_rxresetdone_i : std_logic;
signal gt1_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt1_txprecursor_i : std_logic_vector(4 downto 0);
signal gt1_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt1_txchardispval_i : std_logic_vector(3 downto 0);
signal gt1_txprbsforceerr_i : std_logic;
signal gt1_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt1_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt1_txdata_i : std_logic_vector(31 downto 0);
signal gt1_txoutclkfabric_i : std_logic;
signal gt1_txoutclkpcs_i : std_logic;
signal gt1_txcharisk_i : std_logic_vector(3 downto 0);
signal gt1_txresetdone_i : std_logic;
signal gt1_txpolarity_i : std_logic;
signal gt1_txprbssel_i : std_logic_vector(2 downto 0);


signal gt2_txfsmresetdone_i : std_logic;
signal gt2_rxfsmresetdone_i : std_logic;
signal gt2_track_data_i : std_logic;
signal gt2_txusrclk_i : std_logic;
signal gt2_txusrclk2_i : std_logic;
signal gt2_rxusrclk_i : std_logic;
signal gt2_rxusrclk2_i : std_logic;
signal gt2_cpllfbclklost_i : std_logic;
signal gt2_cplllock_i : std_logic;
signal gt2_drpaddr_i : std_logic_vector(8 downto 0);
signal gt2_drpdi_i : std_logic_vector(15 downto 0);
signal gt2_drpdo_i : std_logic_vector(15 downto 0);
signal gt2_drpen_i : std_logic;
signal gt2_drprdy_i : std_logic;
signal gt2_drpwe_i : std_logic;
signal gt2_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt2_rxpd_i : std_logic_vector(1 downto 0);
signal gt2_txpd_i : std_logic_vector(1 downto 0);
signal gt2_eyescandataerror_i : std_logic;
signal gt2_rxcdrhold_i : std_logic;
signal gt2_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt2_rxdata_i : std_logic_vector(31 downto 0);
signal gt2_rxprbserr_i : std_logic;
signal gt2_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt2_rxprbscntreset_i : std_logic;
signal gt2_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt2_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt2_rxbufreset_i : std_logic;
signal gt2_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt2_rxbyteisaligned_i : std_logic;
signal gt2_rxbyterealign_i : std_logic;
signal gt2_rxcommadet_i : std_logic;
signal gt2_rxmcommaalignen_i : std_logic;
signal gt2_rxpcommaalignen_i : std_logic;
signal gt2_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt2_rxoutclkfabric_i : std_logic;
signal gt2_rxpmareset_i : std_logic;
signal gt2_rxlpmen_i : std_logic;
signal gt2_rxpolarity_i : std_logic;
signal gt2_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt2_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt2_rxresetdone_i : std_logic;
signal gt2_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt2_txprecursor_i : std_logic_vector(4 downto 0);
signal gt2_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt2_txchardispval_i : std_logic_vector(3 downto 0);
signal gt2_txprbsforceerr_i : std_logic;
signal gt2_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt2_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt2_txdata_i : std_logic_vector(31 downto 0);
signal gt2_txoutclkfabric_i : std_logic;
signal gt2_txoutclkpcs_i : std_logic;
signal gt2_txcharisk_i : std_logic_vector(3 downto 0);
signal gt2_txresetdone_i : std_logic;
signal gt2_txpolarity_i : std_logic;
signal gt2_txprbssel_i : std_logic_vector(2 downto 0);


signal gt3_txfsmresetdone_i : std_logic;
signal gt3_rxfsmresetdone_i : std_logic;
signal gt3_track_data_i : std_logic;
signal gt3_txusrclk_i : std_logic;
signal gt3_txusrclk2_i : std_logic;
signal gt3_rxusrclk_i : std_logic;
signal gt3_rxusrclk2_i : std_logic;
signal gt3_cpllfbclklost_i : std_logic;
signal gt3_cplllock_i : std_logic;
signal gt3_drpaddr_i : std_logic_vector(8 downto 0);
signal gt3_drpdi_i : std_logic_vector(15 downto 0);
signal gt3_drpdo_i : std_logic_vector(15 downto 0);
signal gt3_drpen_i : std_logic;
signal gt3_drprdy_i : std_logic;
signal gt3_drpwe_i : std_logic;
signal gt3_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt3_rxpd_i : std_logic_vector(1 downto 0);
signal gt3_txpd_i : std_logic_vector(1 downto 0);
signal gt3_eyescandataerror_i : std_logic;
signal gt3_rxcdrhold_i : std_logic;
signal gt3_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt3_rxdata_i : std_logic_vector(31 downto 0);
signal gt3_rxprbserr_i : std_logic;
signal gt3_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt3_rxprbscntreset_i : std_logic;
signal gt3_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt3_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt3_rxbufreset_i : std_logic;
signal gt3_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt3_rxbyteisaligned_i : std_logic;
signal gt3_rxbyterealign_i : std_logic;
signal gt3_rxcommadet_i : std_logic;
signal gt3_rxmcommaalignen_i : std_logic;
signal gt3_rxpcommaalignen_i : std_logic;
signal gt3_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt3_rxoutclkfabric_i : std_logic;
signal gt3_rxpmareset_i : std_logic;
signal gt3_rxlpmen_i : std_logic;
signal gt3_rxpolarity_i : std_logic;
signal gt3_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt3_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt3_rxresetdone_i : std_logic;
signal gt3_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt3_txprecursor_i : std_logic_vector(4 downto 0);
signal gt3_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt3_txchardispval_i : std_logic_vector(3 downto 0);
signal gt3_txprbsforceerr_i : std_logic;
signal gt3_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt3_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt3_txdata_i : std_logic_vector(31 downto 0);
signal gt3_txoutclkfabric_i : std_logic;
signal gt3_txoutclkpcs_i : std_logic;
signal gt3_txcharisk_i : std_logic_vector(3 downto 0);
signal gt3_txresetdone_i : std_logic;
signal gt3_txpolarity_i : std_logic;
signal gt3_txprbssel_i : std_logic_vector(2 downto 0);


signal gt4_txfsmresetdone_i : std_logic;
signal gt4_rxfsmresetdone_i : std_logic;
signal gt4_track_data_i : std_logic;
signal gt4_txusrclk_i : std_logic;
signal gt4_txusrclk2_i : std_logic;
signal gt4_rxusrclk_i : std_logic;
signal gt4_rxusrclk2_i : std_logic;
signal gt4_cpllfbclklost_i : std_logic;
signal gt4_cplllock_i : std_logic;
signal gt4_drpaddr_i : std_logic_vector(8 downto 0);
signal gt4_drpdi_i : std_logic_vector(15 downto 0);
signal gt4_drpdo_i : std_logic_vector(15 downto 0);
signal gt4_drpen_i : std_logic;
signal gt4_drprdy_i : std_logic;
signal gt4_drpwe_i : std_logic;
signal gt4_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt4_rxpd_i : std_logic_vector(1 downto 0);
signal gt4_txpd_i : std_logic_vector(1 downto 0);
signal gt4_eyescandataerror_i : std_logic;
signal gt4_rxcdrhold_i : std_logic;
signal gt4_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt4_rxdata_i : std_logic_vector(31 downto 0);
signal gt4_rxprbserr_i : std_logic;
signal gt4_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt4_rxprbscntreset_i : std_logic;
signal gt4_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt4_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt4_rxbufreset_i : std_logic;
signal gt4_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt4_rxbyteisaligned_i : std_logic;
signal gt4_rxbyterealign_i : std_logic;
signal gt4_rxcommadet_i : std_logic;
signal gt4_rxmcommaalignen_i : std_logic;
signal gt4_rxpcommaalignen_i : std_logic;
signal gt4_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt4_rxoutclkfabric_i : std_logic;
signal gt4_rxpmareset_i : std_logic;
signal gt4_rxlpmen_i : std_logic;
signal gt4_rxpolarity_i : std_logic;
signal gt4_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt4_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt4_rxresetdone_i : std_logic;
signal gt4_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt4_txprecursor_i : std_logic_vector(4 downto 0);
signal gt4_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt4_txchardispval_i : std_logic_vector(3 downto 0);
signal gt4_txprbsforceerr_i : std_logic;
signal gt4_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt4_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt4_txdata_i : std_logic_vector(31 downto 0);
signal gt4_txoutclkfabric_i : std_logic;
signal gt4_txoutclkpcs_i : std_logic;
signal gt4_txcharisk_i : std_logic_vector(3 downto 0);
signal gt4_txresetdone_i : std_logic;
signal gt4_txpolarity_i : std_logic;
signal gt4_txprbssel_i : std_logic_vector(2 downto 0);


signal gt5_txfsmresetdone_i : std_logic;
signal gt5_rxfsmresetdone_i : std_logic;
signal gt5_track_data_i : std_logic;
signal gt5_txusrclk_i : std_logic;
signal gt5_txusrclk2_i : std_logic;
signal gt5_rxusrclk_i : std_logic;
signal gt5_rxusrclk2_i : std_logic;
signal gt5_cpllfbclklost_i : std_logic;
signal gt5_cplllock_i : std_logic;
signal gt5_drpaddr_i : std_logic_vector(8 downto 0);
signal gt5_drpdi_i : std_logic_vector(15 downto 0);
signal gt5_drpdo_i : std_logic_vector(15 downto 0);
signal gt5_drpen_i : std_logic;
signal gt5_drprdy_i : std_logic;
signal gt5_drpwe_i : std_logic;
signal gt5_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt5_rxpd_i : std_logic_vector(1 downto 0);
signal gt5_txpd_i : std_logic_vector(1 downto 0);
signal gt5_eyescandataerror_i : std_logic;
signal gt5_rxcdrhold_i : std_logic;
signal gt5_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt5_rxdata_i : std_logic_vector(31 downto 0);
signal gt5_rxprbserr_i : std_logic;
signal gt5_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt5_rxprbscntreset_i : std_logic;
signal gt5_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt5_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt5_rxbufreset_i : std_logic;
signal gt5_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt5_rxbyteisaligned_i : std_logic;
signal gt5_rxbyterealign_i : std_logic;
signal gt5_rxcommadet_i : std_logic;
signal gt5_rxmcommaalignen_i : std_logic;
signal gt5_rxpcommaalignen_i : std_logic;
signal gt5_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt5_rxoutclkfabric_i : std_logic;
signal gt5_rxpmareset_i : std_logic;
signal gt5_rxlpmen_i : std_logic;
signal gt5_rxpolarity_i : std_logic;
signal gt5_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt5_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt5_rxresetdone_i : std_logic;
signal gt5_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt5_txprecursor_i : std_logic_vector(4 downto 0);
signal gt5_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt5_txchardispval_i : std_logic_vector(3 downto 0);
signal gt5_txprbsforceerr_i : std_logic;
signal gt5_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt5_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt5_txdata_i : std_logic_vector(31 downto 0);
signal gt5_txoutclkfabric_i : std_logic;
signal gt5_txoutclkpcs_i : std_logic;
signal gt5_txcharisk_i : std_logic_vector(3 downto 0);
signal gt5_txresetdone_i : std_logic;
signal gt5_txpolarity_i : std_logic;
signal gt5_txprbssel_i : std_logic_vector(2 downto 0);


signal gt6_txfsmresetdone_i : std_logic;
signal gt6_rxfsmresetdone_i : std_logic;
signal gt6_track_data_i : std_logic;
signal gt6_txusrclk_i : std_logic;
signal gt6_txusrclk2_i : std_logic;
signal gt6_rxusrclk_i : std_logic;
signal gt6_rxusrclk2_i : std_logic;
signal gt6_cpllfbclklost_i : std_logic;
signal gt6_cplllock_i : std_logic;
signal gt6_drpaddr_i : std_logic_vector(8 downto 0);
signal gt6_drpdi_i : std_logic_vector(15 downto 0);
signal gt6_drpdo_i : std_logic_vector(15 downto 0);
signal gt6_drpen_i : std_logic;
signal gt6_drprdy_i : std_logic;
signal gt6_drpwe_i : std_logic;
signal gt6_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt6_rxpd_i : std_logic_vector(1 downto 0);
signal gt6_txpd_i : std_logic_vector(1 downto 0);
signal gt6_eyescandataerror_i : std_logic;
signal gt6_rxcdrhold_i : std_logic;
signal gt6_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt6_rxdata_i : std_logic_vector(31 downto 0);
signal gt6_rxprbserr_i : std_logic;
signal gt6_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt6_rxprbscntreset_i : std_logic;
signal gt6_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt6_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt6_rxbufreset_i : std_logic;
signal gt6_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt6_rxbyteisaligned_i : std_logic;
signal gt6_rxbyterealign_i : std_logic;
signal gt6_rxcommadet_i : std_logic;
signal gt6_rxmcommaalignen_i : std_logic;
signal gt6_rxpcommaalignen_i : std_logic;
signal gt6_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt6_rxoutclkfabric_i : std_logic;
signal gt6_rxpmareset_i : std_logic;
signal gt6_rxlpmen_i : std_logic;
signal gt6_rxpolarity_i : std_logic;
signal gt6_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt6_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt6_rxresetdone_i : std_logic;
signal gt6_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt6_txprecursor_i : std_logic_vector(4 downto 0);
signal gt6_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt6_txchardispval_i : std_logic_vector(3 downto 0);
signal gt6_txprbsforceerr_i : std_logic;
signal gt6_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt6_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt6_txdata_i : std_logic_vector(31 downto 0);
signal gt6_txoutclkfabric_i : std_logic;
signal gt6_txoutclkpcs_i : std_logic;
signal gt6_txcharisk_i : std_logic_vector(3 downto 0);
signal gt6_txresetdone_i : std_logic;
signal gt6_txpolarity_i : std_logic;
signal gt6_txprbssel_i : std_logic_vector(2 downto 0);


signal gt7_txfsmresetdone_i : std_logic;
signal gt7_rxfsmresetdone_i : std_logic;
signal gt7_track_data_i : std_logic;
signal gt7_txusrclk_i : std_logic;
signal gt7_txusrclk2_i : std_logic;
signal gt7_rxusrclk_i : std_logic;
signal gt7_rxusrclk2_i : std_logic;
signal gt7_cpllfbclklost_i : std_logic;
signal gt7_cplllock_i : std_logic;
signal gt7_drpaddr_i : std_logic_vector(8 downto 0);
signal gt7_drpdi_i : std_logic_vector(15 downto 0);
signal gt7_drpdo_i : std_logic_vector(15 downto 0);
signal gt7_drpen_i : std_logic;
signal gt7_drprdy_i : std_logic;
signal gt7_drpwe_i : std_logic;
signal gt7_dmonitorout_i : std_logic_vector(7 downto 0);
signal gt7_rxpd_i : std_logic_vector(1 downto 0);
signal gt7_txpd_i : std_logic_vector(1 downto 0);
signal gt7_eyescandataerror_i : std_logic;
signal gt7_rxcdrhold_i : std_logic;
signal gt7_rxclkcorcnt_i : std_logic_vector(1 downto 0);
signal gt7_rxdata_i : std_logic_vector(31 downto 0);
signal gt7_rxprbserr_i : std_logic;
signal gt7_rxprbssel_i : std_logic_vector(2 downto 0);
signal gt7_rxprbscntreset_i : std_logic;
signal gt7_rxdisperr_i : std_logic_vector(3 downto 0);
signal gt7_rxnotintable_i : std_logic_vector(3 downto 0);
signal gt7_rxbufreset_i : std_logic;
signal gt7_rxbufstatus_i : std_logic_vector(2 downto 0);
signal gt7_rxbyteisaligned_i : std_logic;
signal gt7_rxbyterealign_i : std_logic;
signal gt7_rxcommadet_i : std_logic;
signal gt7_rxmcommaalignen_i : std_logic;
signal gt7_rxpcommaalignen_i : std_logic;
signal gt7_rxmonitorout_i : std_logic_vector(6 downto 0);
signal gt7_rxoutclkfabric_i : std_logic;
signal gt7_rxpmareset_i : std_logic;
signal gt7_rxlpmen_i : std_logic;
signal gt7_rxpolarity_i : std_logic;
signal gt7_rxchariscomma_i : std_logic_vector(3 downto 0);
signal gt7_rxcharisk_i : std_logic_vector(3 downto 0);
signal gt7_rxresetdone_i : std_logic;
signal gt7_txpostcursor_i : std_logic_vector(4 downto 0);
signal gt7_txprecursor_i : std_logic_vector(4 downto 0);
signal gt7_txchardispmode_i : std_logic_vector(3 downto 0);
signal gt7_txchardispval_i : std_logic_vector(3 downto 0);
signal gt7_txprbsforceerr_i : std_logic;
signal gt7_txbufstatus_i : std_logic_vector(1 downto 0);
signal gt7_txdiffctrl_i : std_logic_vector(3 downto 0);
signal gt7_txdata_i : std_logic_vector(31 downto 0);
signal gt7_txoutclkfabric_i : std_logic;
signal gt7_txoutclkpcs_i : std_logic;
signal gt7_txcharisk_i : std_logic_vector(3 downto 0);
signal gt7_txresetdone_i : std_logic;
signal gt7_txpolarity_i : std_logic;
signal gt7_txprbssel_i : std_logic_vector(2 downto 0);


signal fifo_out : std_logic_vector(31 downto 0);
signal fifo_wr : std_logic;
signal fifo_rd : std_logic;
signal fifo_in : std_logic_vector(31 downto 0);
signal fifo_buf : std_logic_vector(31 downto 0);
signal txdata_out : std_logic_vector(31 downto 0);
signal slaves_rst : std_logic;
signal sys_rst : std_logic;
signal calibrate_firefly : std_logic;

signal rxfsmresetdone_i : std_logic;
signal track_data_i : std_logic;
signal txcharisk_i : std_logic_vector(3 downto 0);

signal mgt_adrs : std_logic_vector(2 downto 0);
signal data_valid_i : std_logic;
signal mgt_mask : std_logic_vector(7 downto 0);

--signal gt0_loopback_i : std_logic_vector(__size downto 0);

begin

--      DCM clock generation for internal bus, ethernet

        clocks: entity work.clocks_7s_extphy
                port map(
                        sysclk_p => sysclk_p,
                        sysclk_n => sysclk_n,
                        clko_125 => clk125,
                        clko_125_90 => clk125_90,
                        clko_200 => clk200,
                        clko_ipb => ipb_clk,
                        locked => locked,
                        nuke => sys_rst,
                        soft_rst => '0',
                        rsto_125 => rst_125,
                        rsto_ipb => rst_ipb,
                        rsto_ipb_ctrl => rst_ipb_ctrl,
                        onehz => onehz
                );
                
--         i2cmaster: entity work.i2c_master
--         	generic map(
--         		input_clk => input_clk,
--         		bus_clk   => bus_clk
--         	)
--         	port map(
--         		clk       => clk200,
--         		reset_n   => rstb,
--         		ena       => ena,
--         		addr      => i2c_addrs,
--         		rw        => i2c_rw,
--         		data_wr   => i2c_data_wr,
--         		busy      => busy,
--         		data_rd   => i2c_data_rd,
--         		ack_error => i2c_ackerror,
--         		sda       => SDA,
--         		scl       => SCL
--         	);
         	
         	
--        ena_ctrl: process(clk200)
--		begin
--			if rising_edge(clk200) then
--				if (rstb = '0') then
--					prevbusy <= '0';
--					prevgo <= '0';
--					ena <= '0';
--					tempena <= '0';
--				else
--					if (prevbusy = '1' and busy = '0') then
--						if (tempena='1') then 
--							ena <= '1';
--							tempena <= '0';
--						else ena <= '0';
--						end if;
--					end if;
--					
--					if (prevgo = '0' and go = '1' and busy = '1')  then
--						tempena <= '1'; --this covers the instance when the go comes in while busy is high.
--					elsif (prevgo = '0' and go = '1' and busy = '0')  then
--						ena <= '1';
--					end if;
--					
--					prevbusy <= busy;
--					prevgo <= go;
--
--					
--				end if;
--			end if;
--		end process ena_ctrl;
                
--       clocks: entity work.clocks_7s_extphy_se
--		port map(
--			sysclk_p => sysclk_p,
--			sysclk_n => sysclk_n,
--			clko_125 => clk125,
--			clko_125_90 => clk125_90,
--			clko_200 => clk200,
--			clko_ipb => clk_ipb_i,
--			locked => locked,
--			nuke => nuke,
--			soft_rst => soft_rst,
--			rsto_125 => rst125,
--			rsto_ipb => rst_ipb,
--			rsto_ipb_ctrl => rst_ipb_ctrl,
--			onehz => onehz
--		);

        leds <= (fifo_rd, '1',led_ipbus(0),led_ipbus(1), '0', '0', locked, onehz);
		

		tied_to_ground_i                             <= '0';
		tied_to_vcc_i								<= '1';
    gtx_reset_firefly_support_i : gtx_reset_firefly
    port map
    (
        SOFT_RESET_TX_IN                =>      soft_rst,
--        SOFT_RESET_RX_IN                =>      soft_rst,
        DONT_RESET_ON_DATA_ERROR_IN     =>      tied_to_ground_i,
	    Q1_CLK0_GTREFCLK_PAD_N_IN 		=> 		Q1_CLK0_GTREFCLK_PAD_N_IN,
	    Q1_CLK0_GTREFCLK_PAD_P_IN 		=> 		Q1_CLK0_GTREFCLK_PAD_P_IN,
	    --Q1_CLK1_GTREFCLK_PAD_N_IN 		=> 		Q1_CLK1_GTREFCLK_PAD_N_IN,
	    --Q1_CLK1_GTREFCLK_PAD_P_IN 		=> 		Q1_CLK1_GTREFCLK_PAD_P_IN,
        GT0_TX_FSM_RESET_DONE_OUT       =>      gt0_txfsmresetdone_i,
        GT0_RX_FSM_RESET_DONE_OUT       =>      gt0_rxfsmresetdone_i,
        GT0_DATA_VALID_IN               =>      gt0_track_data_i,
        GT1_TX_FSM_RESET_DONE_OUT       =>      gt1_txfsmresetdone_i,
        GT1_RX_FSM_RESET_DONE_OUT       =>      gt1_rxfsmresetdone_i,
        GT1_DATA_VALID_IN               =>      gt1_track_data_i,
        GT2_TX_FSM_RESET_DONE_OUT       =>      gt2_txfsmresetdone_i,
        GT2_RX_FSM_RESET_DONE_OUT       =>      gt2_rxfsmresetdone_i,
        GT2_DATA_VALID_IN               =>      gt2_track_data_i,
        GT3_TX_FSM_RESET_DONE_OUT       =>      gt3_txfsmresetdone_i,
        GT3_RX_FSM_RESET_DONE_OUT       =>      gt3_rxfsmresetdone_i,
        GT3_DATA_VALID_IN               =>      gt3_track_data_i,
        GT4_TX_FSM_RESET_DONE_OUT       =>      gt4_txfsmresetdone_i,
        GT4_RX_FSM_RESET_DONE_OUT       =>      gt4_rxfsmresetdone_i,
        GT4_DATA_VALID_IN               =>      gt4_track_data_i,
        GT5_TX_FSM_RESET_DONE_OUT       =>      gt5_txfsmresetdone_i,
        GT5_RX_FSM_RESET_DONE_OUT       =>      gt5_rxfsmresetdone_i,
        GT5_DATA_VALID_IN               =>      gt5_track_data_i,
        GT6_TX_FSM_RESET_DONE_OUT       =>      gt6_txfsmresetdone_i,
        GT6_RX_FSM_RESET_DONE_OUT       =>      gt6_rxfsmresetdone_i,
        GT6_DATA_VALID_IN               =>      gt6_track_data_i,
        GT7_TX_FSM_RESET_DONE_OUT       =>      gt7_txfsmresetdone_i,
        GT7_RX_FSM_RESET_DONE_OUT       =>      gt7_rxfsmresetdone_i,
        GT7_DATA_VALID_IN               =>      gt7_track_data_i,
 
    GT0_TXUSRCLK_OUT => gt0_txusrclk_i,
    GT0_TXUSRCLK2_OUT => gt0_txusrclk2_i,
--    GT0_RXUSRCLK_OUT => gt0_rxusrclk_i,
--    GT0_RXUSRCLK2_OUT => gt0_rxusrclk2_i,
 
    GT1_TXUSRCLK_OUT => gt1_txusrclk_i,
    GT1_TXUSRCLK2_OUT => gt1_txusrclk2_i,
--    GT1_RXUSRCLK_OUT => gt1_rxusrclk_i,
--    GT1_RXUSRCLK2_OUT => gt1_rxusrclk2_i,
 
    GT2_TXUSRCLK_OUT => gt2_txusrclk_i,
    GT2_TXUSRCLK2_OUT => gt2_txusrclk2_i,
--    GT2_RXUSRCLK_OUT => gt2_rxusrclk_i,
--    GT2_RXUSRCLK2_OUT => gt2_rxusrclk2_i,
 
    GT3_TXUSRCLK_OUT => gt3_txusrclk_i,
    GT3_TXUSRCLK2_OUT => gt3_txusrclk2_i,
--    GT3_RXUSRCLK_OUT => gt3_rxusrclk_i,
--    GT3_RXUSRCLK2_OUT => gt3_rxusrclk2_i,
 
    GT4_TXUSRCLK_OUT => gt4_txusrclk_i,
    GT4_TXUSRCLK2_OUT => gt4_txusrclk2_i,
--    GT4_RXUSRCLK_OUT => gt4_rxusrclk_i,
--    GT4_RXUSRCLK2_OUT => gt4_rxusrclk2_i,
 
    GT5_TXUSRCLK_OUT => gt5_txusrclk_i,
    GT5_TXUSRCLK2_OUT => gt5_txusrclk2_i,
--    GT5_RXUSRCLK_OUT => gt5_rxusrclk_i,
--    GT5_RXUSRCLK2_OUT => gt5_rxusrclk2_i,
 
    GT6_TXUSRCLK_OUT => gt6_txusrclk_i,
    GT6_TXUSRCLK2_OUT => gt6_txusrclk2_i,
--    GT6_RXUSRCLK_OUT => gt6_rxusrclk_i,
--    GT6_RXUSRCLK2_OUT => gt6_rxusrclk2_i,
 
    GT7_TXUSRCLK_OUT => gt7_txusrclk_i,
    GT7_TXUSRCLK2_OUT => gt7_txusrclk2_i,
--    GT7_RXUSRCLK_OUT => gt7_rxusrclk_i,
--    GT7_RXUSRCLK2_OUT => gt7_rxusrclk2_i,

 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT0  (X1Y0)

        --------------------------------- CPLL Ports -------------------------------
        gt0_cpllfbclklost_out           =>      gt0_cpllfbclklost_i,
        gt0_cplllock_out                =>      gt0_cplllock_i,
        gt0_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt0_drpaddr_in                  =>      gt0_drpaddr_i,
        gt0_drpdi_in                    =>      gt0_drpdi_i,
        gt0_drpdo_out                   =>      gt0_drpdo_i,
        gt0_drpen_in                    =>      gt0_drpen_i,
        gt0_drprdy_out                  =>      gt0_drprdy_i,
        gt0_drpwe_in                    =>      gt0_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt0_dmonitorout_out             =>      gt0_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt0_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt0_rxpd_in                     =>      gt0_rxpd_i,
        gt0_txpd_in                     =>      gt0_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt0_eyescanreset_in             =>      tied_to_ground_i,
--        gt0_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt0_eyescandataerror_out        =>      gt0_eyescandataerror_i,
        gt0_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt0_rxcdrhold_in                =>      gt0_rxcdrhold_i,
        gt0_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt0_rxclkcorcnt_out             =>      gt0_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt0_rxdata_out                  =>      gt0_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt0_rxprbserr_out               =>      gt0_rxprbserr_i,
        gt0_rxprbssel_in                =>      gt0_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt0_rxprbscntreset_in           =>      gt0_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt0_rxdisperr_out               =>      gt0_rxdisperr_i,
--        gt0_rxnotintable_out            =>      gt0_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt0_gtxrxp_in                   =>      RXP_IN(0),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt0_gtxrxn_in                   =>      RXN_IN(0),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt0_rxbufreset_in               =>      gt0_rxbufreset_i,
        gt0_rxbufstatus_out             =>      gt0_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt0_rxbyteisaligned_out         =>      gt0_rxbyteisaligned_i,
--        gt0_rxbyterealign_out           =>      gt0_rxbyterealign_i,
--        gt0_rxcommadet_out              =>      gt0_rxcommadet_i,
--        gt0_rxmcommaalignen_in          =>      gt0_rxmcommaalignen_i,
--        gt0_rxpcommaalignen_in          =>      gt0_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt0_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt0_rxmonitorout_out            =>      gt0_rxmonitorout_i,
        gt0_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt0_rxoutclkfabric_out          =>      gt0_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt0_gtrxreset_in                =>      tied_to_ground_i,
        gt0_rxpcsreset_in               =>      tied_to_ground_i,
--        gt0_rxpmareset_in               =>      gt0_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt0_rxlpmen_in                  =>      gt0_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt0_rxpolarity_in               =>      gt0_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt0_rxchariscomma_out           =>      gt0_rxchariscomma_i,
--        gt0_rxcharisk_out               =>      gt0_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt0_rxresetdone_out             =>      gt0_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt0_txpostcursor_in             =>      gt0_txpostcursor_i,
        gt0_txprecursor_in              =>      gt0_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt0_gttxreset_in                =>      tied_to_ground_i,
        gt0_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt0_txchardispmode_in           =>      gt0_txchardispmode_i,
        gt0_txchardispval_in            =>      gt0_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt0_txprbsforceerr_in           =>      gt0_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt0_txbufstatus_out             =>      gt0_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt0_txdiffctrl_in               =>      gt0_txdiffctrl_i,
        gt0_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt0_txdata_in                   =>      gt0_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt0_gtxtxn_out                  =>      TXN_OUT(0),
        gt0_gtxtxp_out                  =>      TXP_OUT(0),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt0_txoutclkfabric_out          =>      gt0_txoutclkfabric_i,
        gt0_txoutclkpcs_out             =>      gt0_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt0_txcharisk_in                =>      gt0_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt0_txpcsreset_in               =>      tied_to_ground_i,
        gt0_txpmareset_in               =>      tied_to_ground_i,
        gt0_txresetdone_out             =>      gt0_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt0_txpolarity_in               =>      gt0_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt0_txprbssel_in                =>      gt0_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT1  (X1Y1)

        --------------------------------- CPLL Ports -------------------------------
        gt1_cpllfbclklost_out           =>      gt1_cpllfbclklost_i,
        gt1_cplllock_out                =>      gt1_cplllock_i,
        gt1_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt1_drpaddr_in                  =>      gt1_drpaddr_i,
        gt1_drpdi_in                    =>      gt1_drpdi_i,
        gt1_drpdo_out                   =>      gt1_drpdo_i,
        gt1_drpen_in                    =>      gt1_drpen_i,
        gt1_drprdy_out                  =>      gt1_drprdy_i,
        gt1_drpwe_in                    =>      gt1_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt1_dmonitorout_out             =>      gt1_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt1_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt1_rxpd_in                     =>      gt1_rxpd_i,
        gt1_txpd_in                     =>      gt1_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt1_eyescanreset_in             =>      tied_to_ground_i,
--        gt1_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt1_eyescandataerror_out        =>      gt1_eyescandataerror_i,
        gt1_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt1_rxcdrhold_in                =>      gt1_rxcdrhold_i,
        gt1_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt1_rxclkcorcnt_out             =>      gt1_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt1_rxdata_out                  =>      gt1_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt1_rxprbserr_out               =>      gt1_rxprbserr_i,
        gt1_rxprbssel_in                =>      gt1_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt1_rxprbscntreset_in           =>      gt1_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt1_rxdisperr_out               =>      gt1_rxdisperr_i,
--        gt1_rxnotintable_out            =>      gt1_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt1_gtxrxp_in                   =>      RXP_IN(1),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt1_gtxrxn_in                   =>      RXN_IN(1),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt1_rxbufreset_in               =>      gt1_rxbufreset_i,
        gt1_rxbufstatus_out             =>      gt1_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt1_rxbyteisaligned_out         =>      gt1_rxbyteisaligned_i,
--        gt1_rxbyterealign_out           =>      gt1_rxbyterealign_i,
--        gt1_rxcommadet_out              =>      gt1_rxcommadet_i,
--        gt1_rxmcommaalignen_in          =>      gt1_rxmcommaalignen_i,
--        gt1_rxpcommaalignen_in          =>      gt1_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt1_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt1_rxmonitorout_out            =>      gt1_rxmonitorout_i,
        gt1_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt1_rxoutclkfabric_out          =>      gt1_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt1_gtrxreset_in                =>      tied_to_ground_i,
        gt1_rxpcsreset_in               =>      tied_to_ground_i,
--        gt1_rxpmareset_in               =>      gt1_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt1_rxlpmen_in                  =>      gt1_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt1_rxpolarity_in               =>      gt1_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt1_rxchariscomma_out           =>      gt1_rxchariscomma_i,
--        gt1_rxcharisk_out               =>      gt1_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt1_rxresetdone_out             =>      gt1_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt1_txpostcursor_in             =>      gt1_txpostcursor_i,
        gt1_txprecursor_in              =>      gt1_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt1_gttxreset_in                =>      tied_to_ground_i,
        gt1_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt1_txchardispmode_in           =>      gt1_txchardispmode_i,
        gt1_txchardispval_in            =>      gt1_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt1_txprbsforceerr_in           =>      gt1_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt1_txbufstatus_out             =>      gt1_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt1_txdiffctrl_in               =>      gt1_txdiffctrl_i,
        gt1_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt1_txdata_in                   =>      gt1_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt1_gtxtxn_out                  =>      TXN_OUT(1),
        gt1_gtxtxp_out                  =>      TXP_OUT(1),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt1_txoutclkfabric_out          =>      gt1_txoutclkfabric_i,
        gt1_txoutclkpcs_out             =>      gt1_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt1_txcharisk_in                =>      gt1_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt1_txpcsreset_in               =>      tied_to_ground_i,
        gt1_txpmareset_in               =>      tied_to_ground_i,
        gt1_txresetdone_out             =>      gt1_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt1_txpolarity_in               =>      gt1_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt1_txprbssel_in                =>      gt1_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT2  (X1Y2)

        --------------------------------- CPLL Ports -------------------------------
        gt2_cpllfbclklost_out           =>      gt2_cpllfbclklost_i,
        gt2_cplllock_out                =>      gt2_cplllock_i,
        gt2_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt2_drpaddr_in                  =>      gt2_drpaddr_i,
        gt2_drpdi_in                    =>      gt2_drpdi_i,
        gt2_drpdo_out                   =>      gt2_drpdo_i,
        gt2_drpen_in                    =>      gt2_drpen_i,
        gt2_drprdy_out                  =>      gt2_drprdy_i,
        gt2_drpwe_in                    =>      gt2_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt2_dmonitorout_out             =>      gt2_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt2_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt2_rxpd_in                     =>      gt2_rxpd_i,
        gt2_txpd_in                     =>      gt2_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt2_eyescanreset_in             =>      tied_to_ground_i,
--        gt2_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt2_eyescandataerror_out        =>      gt2_eyescandataerror_i,
        gt2_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt2_rxcdrhold_in                =>      gt2_rxcdrhold_i,
        gt2_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt2_rxclkcorcnt_out             =>      gt2_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt2_rxdata_out                  =>      gt2_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt2_rxprbserr_out               =>      gt2_rxprbserr_i,
        gt2_rxprbssel_in                =>      gt2_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt2_rxprbscntreset_in           =>      gt2_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt2_rxdisperr_out               =>      gt2_rxdisperr_i,
--        gt2_rxnotintable_out            =>      gt2_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt2_gtxrxp_in                   =>      RXP_IN(2),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt2_gtxrxn_in                   =>      RXN_IN(2),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt2_rxbufreset_in               =>      gt2_rxbufreset_i,
        gt2_rxbufstatus_out             =>      gt2_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt2_rxbyteisaligned_out         =>      gt2_rxbyteisaligned_i,
--        gt2_rxbyterealign_out           =>      gt2_rxbyterealign_i,
--        gt2_rxcommadet_out              =>      gt2_rxcommadet_i,
--        gt2_rxmcommaalignen_in          =>      gt2_rxmcommaalignen_i,
--        gt2_rxpcommaalignen_in          =>      gt2_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt2_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt2_rxmonitorout_out            =>      gt2_rxmonitorout_i,
        gt2_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt2_rxoutclkfabric_out          =>      gt2_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt2_gtrxreset_in                =>      tied_to_ground_i,
        gt2_rxpcsreset_in               =>      tied_to_ground_i,
--        gt2_rxpmareset_in               =>      gt2_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt2_rxlpmen_in                  =>      gt2_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt2_rxpolarity_in               =>      gt2_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt2_rxchariscomma_out           =>      gt2_rxchariscomma_i,
--        gt2_rxcharisk_out               =>      gt2_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt2_rxresetdone_out             =>      gt2_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt2_txpostcursor_in             =>      gt2_txpostcursor_i,
        gt2_txprecursor_in              =>      gt2_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt2_gttxreset_in                =>      tied_to_ground_i,
        gt2_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt2_txchardispmode_in           =>      gt2_txchardispmode_i,
        gt2_txchardispval_in            =>      gt2_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt2_txprbsforceerr_in           =>      gt2_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt2_txbufstatus_out             =>      gt2_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt2_txdiffctrl_in               =>      gt2_txdiffctrl_i,
        gt2_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt2_txdata_in                   =>      gt2_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt2_gtxtxn_out                  =>      TXN_OUT(2),
        gt2_gtxtxp_out                  =>      TXP_OUT(2),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt2_txoutclkfabric_out          =>      gt2_txoutclkfabric_i,
        gt2_txoutclkpcs_out             =>      gt2_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt2_txcharisk_in                =>      gt2_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt2_txpcsreset_in               =>      tied_to_ground_i,
        gt2_txpmareset_in               =>      tied_to_ground_i,
        gt2_txresetdone_out             =>      gt2_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt2_txpolarity_in               =>      gt2_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt2_txprbssel_in                =>      gt2_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT3  (X1Y3)

        --------------------------------- CPLL Ports -------------------------------
        gt3_cpllfbclklost_out           =>      gt3_cpllfbclklost_i,
        gt3_cplllock_out                =>      gt3_cplllock_i,
        gt3_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt3_drpaddr_in                  =>      gt3_drpaddr_i,
        gt3_drpdi_in                    =>      gt3_drpdi_i,
        gt3_drpdo_out                   =>      gt3_drpdo_i,
        gt3_drpen_in                    =>      gt3_drpen_i,
        gt3_drprdy_out                  =>      gt3_drprdy_i,
        gt3_drpwe_in                    =>      gt3_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt3_dmonitorout_out             =>      gt3_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt3_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt3_rxpd_in                     =>      gt3_rxpd_i,
        gt3_txpd_in                     =>      gt3_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt3_eyescanreset_in             =>      tied_to_ground_i,
--        gt3_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt3_eyescandataerror_out        =>      gt3_eyescandataerror_i,
        gt3_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt3_rxcdrhold_in                =>      gt3_rxcdrhold_i,
        gt3_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt3_rxclkcorcnt_out             =>      gt3_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt3_rxdata_out                  =>      gt3_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt3_rxprbserr_out               =>      gt3_rxprbserr_i,
        gt3_rxprbssel_in                =>      gt3_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt3_rxprbscntreset_in           =>      gt3_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt3_rxdisperr_out               =>      gt3_rxdisperr_i,
--        gt3_rxnotintable_out            =>      gt3_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt3_gtxrxp_in                   =>      RXP_IN(3),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt3_gtxrxn_in                   =>      RXN_IN(3),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt3_rxbufreset_in               =>      gt3_rxbufreset_i,
        gt3_rxbufstatus_out             =>      gt3_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt3_rxbyteisaligned_out         =>      gt3_rxbyteisaligned_i,
--        gt3_rxbyterealign_out           =>      gt3_rxbyterealign_i,
--        gt3_rxcommadet_out              =>      gt3_rxcommadet_i,
--        gt3_rxmcommaalignen_in          =>      gt3_rxmcommaalignen_i,
--        gt3_rxpcommaalignen_in          =>      gt3_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt3_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt3_rxmonitorout_out            =>      gt3_rxmonitorout_i,
        gt3_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt3_rxoutclkfabric_out          =>      gt3_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt3_gtrxreset_in                =>      tied_to_ground_i,
        gt3_rxpcsreset_in               =>      tied_to_ground_i,
--        gt3_rxpmareset_in               =>      gt3_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt3_rxlpmen_in                  =>      gt3_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt3_rxpolarity_in               =>      gt3_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt3_rxchariscomma_out           =>      gt3_rxchariscomma_i,
--        gt3_rxcharisk_out               =>      gt3_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt3_rxresetdone_out             =>      gt3_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt3_txpostcursor_in             =>      gt3_txpostcursor_i,
        gt3_txprecursor_in              =>      gt3_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt3_gttxreset_in                =>      tied_to_ground_i,
        gt3_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt3_txchardispmode_in           =>      gt3_txchardispmode_i,
        gt3_txchardispval_in            =>      gt3_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt3_txprbsforceerr_in           =>      gt3_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt3_txbufstatus_out             =>      gt3_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt3_txdiffctrl_in               =>      gt3_txdiffctrl_i,
        gt3_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt3_txdata_in                   =>      gt3_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt3_gtxtxn_out                  =>      TXN_OUT(3),
        gt3_gtxtxp_out                  =>      TXP_OUT(3),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt3_txoutclkfabric_out          =>      gt3_txoutclkfabric_i,
        gt3_txoutclkpcs_out             =>      gt3_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt3_txcharisk_in                =>      gt3_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt3_txpcsreset_in               =>      tied_to_ground_i,
        gt3_txpmareset_in               =>      tied_to_ground_i,
        gt3_txresetdone_out             =>      gt3_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt3_txpolarity_in               =>      gt3_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt3_txprbssel_in                =>      gt3_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT4  (X1Y4)

        --------------------------------- CPLL Ports -------------------------------
        gt4_cpllfbclklost_out           =>      gt4_cpllfbclklost_i,
        gt4_cplllock_out                =>      gt4_cplllock_i,
        gt4_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt4_drpaddr_in                  =>      gt4_drpaddr_i,
        gt4_drpdi_in                    =>      gt4_drpdi_i,
        gt4_drpdo_out                   =>      gt4_drpdo_i,
        gt4_drpen_in                    =>      gt4_drpen_i,
        gt4_drprdy_out                  =>      gt4_drprdy_i,
        gt4_drpwe_in                    =>      gt4_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt4_dmonitorout_out             =>      gt4_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt4_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt4_rxpd_in                     =>      gt4_rxpd_i,
        gt4_txpd_in                     =>      gt4_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt4_eyescanreset_in             =>      tied_to_ground_i,
--        gt4_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt4_eyescandataerror_out        =>      gt4_eyescandataerror_i,
        gt4_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt4_rxcdrhold_in                =>      gt4_rxcdrhold_i,
        gt4_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt4_rxclkcorcnt_out             =>      gt4_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt4_rxdata_out                  =>      gt4_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt4_rxprbserr_out               =>      gt4_rxprbserr_i,
        gt4_rxprbssel_in                =>      gt4_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt4_rxprbscntreset_in           =>      gt4_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt4_rxdisperr_out               =>      gt4_rxdisperr_i,
--        gt4_rxnotintable_out            =>      gt4_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt4_gtxrxp_in                   =>      RXP_IN(4),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt4_gtxrxn_in                   =>      RXN_IN(4),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt4_rxbufreset_in               =>      gt4_rxbufreset_i,
        gt4_rxbufstatus_out             =>      gt4_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt4_rxbyteisaligned_out         =>      gt4_rxbyteisaligned_i,
--        gt4_rxbyterealign_out           =>      gt4_rxbyterealign_i,
--        gt4_rxcommadet_out              =>      gt4_rxcommadet_i,
--        gt4_rxmcommaalignen_in          =>      gt4_rxmcommaalignen_i,
--        gt4_rxpcommaalignen_in          =>      gt4_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt4_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt4_rxmonitorout_out            =>      gt4_rxmonitorout_i,
        gt4_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt4_rxoutclkfabric_out          =>      gt4_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt4_gtrxreset_in                =>      tied_to_ground_i,
        gt4_rxpcsreset_in               =>      tied_to_ground_i,
--        gt4_rxpmareset_in               =>      gt4_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt4_rxlpmen_in                  =>      gt4_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt4_rxpolarity_in               =>      gt4_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt4_rxchariscomma_out           =>      gt4_rxchariscomma_i,
--        gt4_rxcharisk_out               =>      gt4_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt4_rxresetdone_out             =>      gt4_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt4_txpostcursor_in             =>      gt4_txpostcursor_i,
        gt4_txprecursor_in              =>      gt4_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt4_gttxreset_in                =>      tied_to_ground_i,
        gt4_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt4_txchardispmode_in           =>      gt4_txchardispmode_i,
        gt4_txchardispval_in            =>      gt4_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt4_txprbsforceerr_in           =>      gt4_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt4_txbufstatus_out             =>      gt4_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt4_txdiffctrl_in               =>      gt4_txdiffctrl_i,
        gt4_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt4_txdata_in                   =>      gt4_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt4_gtxtxn_out                  =>      TXN_OUT(4),
        gt4_gtxtxp_out                  =>      TXP_OUT(4),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt4_txoutclkfabric_out          =>      gt4_txoutclkfabric_i,
        gt4_txoutclkpcs_out             =>      gt4_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt4_txcharisk_in                =>      gt4_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt4_txpcsreset_in               =>      tied_to_ground_i,
        gt4_txpmareset_in               =>      tied_to_ground_i,
        gt4_txresetdone_out             =>      gt4_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt4_txpolarity_in               =>      gt4_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt4_txprbssel_in                =>      gt4_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT5  (X1Y5)

        --------------------------------- CPLL Ports -------------------------------
        gt5_cpllfbclklost_out           =>      gt5_cpllfbclklost_i,
        gt5_cplllock_out                =>      gt5_cplllock_i,
        gt5_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt5_drpaddr_in                  =>      gt5_drpaddr_i,
        gt5_drpdi_in                    =>      gt5_drpdi_i,
        gt5_drpdo_out                   =>      gt5_drpdo_i,
        gt5_drpen_in                    =>      gt5_drpen_i,
        gt5_drprdy_out                  =>      gt5_drprdy_i,
        gt5_drpwe_in                    =>      gt5_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt5_dmonitorout_out             =>      gt5_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt5_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt5_rxpd_in                     =>      gt5_rxpd_i,
        gt5_txpd_in                     =>      gt5_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt5_eyescanreset_in             =>      tied_to_ground_i,
--        gt5_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt5_eyescandataerror_out        =>      gt5_eyescandataerror_i,
        gt5_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt5_rxcdrhold_in                =>      gt5_rxcdrhold_i,
        gt5_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt5_rxclkcorcnt_out             =>      gt5_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt5_rxdata_out                  =>      gt5_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt5_rxprbserr_out               =>      gt5_rxprbserr_i,
        gt5_rxprbssel_in                =>      gt5_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt5_rxprbscntreset_in           =>      gt5_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt5_rxdisperr_out               =>      gt5_rxdisperr_i,
--        gt5_rxnotintable_out            =>      gt5_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt5_gtxrxp_in                   =>      RXP_IN(5),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt5_gtxrxn_in                   =>      RXN_IN(5),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt5_rxbufreset_in               =>      gt5_rxbufreset_i,
        gt5_rxbufstatus_out             =>      gt5_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt5_rxbyteisaligned_out         =>      gt5_rxbyteisaligned_i,
--        gt5_rxbyterealign_out           =>      gt5_rxbyterealign_i,
--        gt5_rxcommadet_out              =>      gt5_rxcommadet_i,
--        gt5_rxmcommaalignen_in          =>      gt5_rxmcommaalignen_i,
--        gt5_rxpcommaalignen_in          =>      gt5_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt5_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt5_rxmonitorout_out            =>      gt5_rxmonitorout_i,
        gt5_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt5_rxoutclkfabric_out          =>      gt5_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt5_gtrxreset_in                =>      tied_to_ground_i,
        gt5_rxpcsreset_in               =>      tied_to_ground_i,
--        gt5_rxpmareset_in               =>      gt5_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt5_rxlpmen_in                  =>      gt5_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt5_rxpolarity_in               =>      gt5_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt5_rxchariscomma_out           =>      gt5_rxchariscomma_i,
--        gt5_rxcharisk_out               =>      gt5_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt5_rxresetdone_out             =>      gt5_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt5_txpostcursor_in             =>      gt5_txpostcursor_i,
        gt5_txprecursor_in              =>      gt5_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt5_gttxreset_in                =>      tied_to_ground_i,
        gt5_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt5_txchardispmode_in           =>      gt5_txchardispmode_i,
        gt5_txchardispval_in            =>      gt5_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt5_txprbsforceerr_in           =>      gt5_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt5_txbufstatus_out             =>      gt5_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt5_txdiffctrl_in               =>      gt5_txdiffctrl_i,
        gt5_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt5_txdata_in                   =>      gt5_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt5_gtxtxn_out                  =>      TXN_OUT(5),
        gt5_gtxtxp_out                  =>      TXP_OUT(5),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt5_txoutclkfabric_out          =>      gt5_txoutclkfabric_i,
        gt5_txoutclkpcs_out             =>      gt5_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt5_txcharisk_in                =>      gt5_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt5_txpcsreset_in               =>      tied_to_ground_i,
        gt5_txpmareset_in               =>      tied_to_ground_i,
        gt5_txresetdone_out             =>      gt5_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt5_txpolarity_in               =>      gt5_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt5_txprbssel_in                =>      gt5_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT6  (X1Y6)

        --------------------------------- CPLL Ports -------------------------------
        gt6_cpllfbclklost_out           =>      gt6_cpllfbclklost_i,
        gt6_cplllock_out                =>      gt6_cplllock_i,
        gt6_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt6_drpaddr_in                  =>      gt6_drpaddr_i,
        gt6_drpdi_in                    =>      gt6_drpdi_i,
        gt6_drpdo_out                   =>      gt6_drpdo_i,
        gt6_drpen_in                    =>      gt6_drpen_i,
        gt6_drprdy_out                  =>      gt6_drprdy_i,
        gt6_drpwe_in                    =>      gt6_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt6_dmonitorout_out             =>      gt6_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt6_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt6_rxpd_in                     =>      gt6_rxpd_i,
        gt6_txpd_in                     =>      gt6_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt6_eyescanreset_in             =>      tied_to_ground_i,
--        gt6_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt6_eyescandataerror_out        =>      gt6_eyescandataerror_i,
        gt6_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt6_rxcdrhold_in                =>      gt6_rxcdrhold_i,
        gt6_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt6_rxclkcorcnt_out             =>      gt6_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt6_rxdata_out                  =>      gt6_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt6_rxprbserr_out               =>      gt6_rxprbserr_i,
        gt6_rxprbssel_in                =>      gt6_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt6_rxprbscntreset_in           =>      gt6_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt6_rxdisperr_out               =>      gt6_rxdisperr_i,
--        gt6_rxnotintable_out            =>      gt6_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt6_gtxrxp_in                   =>      RXP_IN(6),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt6_gtxrxn_in                   =>      RXN_IN(6),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt6_rxbufreset_in               =>      gt6_rxbufreset_i,
        gt6_rxbufstatus_out             =>      gt6_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt6_rxbyteisaligned_out         =>      gt6_rxbyteisaligned_i,
--        gt6_rxbyterealign_out           =>      gt6_rxbyterealign_i,
--        gt6_rxcommadet_out              =>      gt6_rxcommadet_i,
--        gt6_rxmcommaalignen_in          =>      gt6_rxmcommaalignen_i,
--        gt6_rxpcommaalignen_in          =>      gt6_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt6_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt6_rxmonitorout_out            =>      gt6_rxmonitorout_i,
        gt6_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt6_rxoutclkfabric_out          =>      gt6_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt6_gtrxreset_in                =>      tied_to_ground_i,
        gt6_rxpcsreset_in               =>      tied_to_ground_i,
--        gt6_rxpmareset_in               =>      gt6_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt6_rxlpmen_in                  =>      gt6_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt6_rxpolarity_in               =>      gt6_rxpolarity_i,
        ------------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt6_rxchariscomma_out           =>      gt6_rxchariscomma_i,
--        gt6_rxcharisk_out               =>      gt6_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt6_rxresetdone_out             =>      gt6_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt6_txpostcursor_in             =>      gt6_txpostcursor_i,
        gt6_txprecursor_in              =>      gt6_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt6_gttxreset_in                =>      tied_to_ground_i,
        gt6_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt6_txchardispmode_in           =>      gt6_txchardispmode_i,
        gt6_txchardispval_in            =>      gt6_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt6_txprbsforceerr_in           =>      gt6_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt6_txbufstatus_out             =>      gt6_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt6_txdiffctrl_in               =>      gt6_txdiffctrl_i,
        gt6_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt6_txdata_in                   =>      gt6_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt6_gtxtxn_out                  =>      TXN_OUT(6),
        gt6_gtxtxp_out                  =>      TXP_OUT(6),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt6_txoutclkfabric_out          =>      gt6_txoutclkfabric_i,
        gt6_txoutclkpcs_out             =>      gt6_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt6_txcharisk_in                =>      gt6_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt6_txpcsreset_in               =>      tied_to_ground_i,
        gt6_txpmareset_in               =>      tied_to_ground_i,
        gt6_txresetdone_out             =>      gt6_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt6_txpolarity_in               =>      gt6_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt6_txprbssel_in                =>      gt6_txprbssel_i,


 

        --_____________________________________________________________________
        --_____________________________________________________________________
        --GT7  (X1Y7)

        --------------------------------- CPLL Ports -------------------------------
        gt7_cpllfbclklost_out           =>      gt7_cpllfbclklost_i,
        gt7_cplllock_out                =>      gt7_cplllock_i,
        gt7_cpllreset_in                =>      tied_to_ground_i,
        ---------------------------- Channel - DRP Ports  --------------------------
        gt7_drpaddr_in                  =>      gt7_drpaddr_i,
        gt7_drpdi_in                    =>      gt7_drpdi_i,
        gt7_drpdo_out                   =>      gt7_drpdo_i,
        gt7_drpen_in                    =>      gt7_drpen_i,
        gt7_drprdy_out                  =>      gt7_drprdy_i,
        gt7_drpwe_in                    =>      gt7_drpwe_i,
        --------------------------- Digital Monitor Ports --------------------------
        gt7_dmonitorout_out             =>      gt7_dmonitorout_i,
        ------------------------------- Loopback Ports -----------------------------
        gt7_loopback_in                 =>      "000",
        ------------------------------ Power-Down Ports ----------------------------
        gt7_rxpd_in                     =>      gt7_rxpd_i,
        gt7_txpd_in                     =>      gt7_txpd_i,
        --------------------- RX Initialization and Reset Ports --------------------
        gt7_eyescanreset_in             =>      tied_to_ground_i,
--        gt7_rxuserrdy_in                =>      tied_to_vcc_i,
        -------------------------- RX Margin Analysis Ports ------------------------
        gt7_eyescandataerror_out        =>      gt7_eyescandataerror_i,
        gt7_eyescantrigger_in           =>      tied_to_ground_i,
        ------------------------- Receive Ports - CDR Ports ------------------------
--        gt7_rxcdrhold_in                =>      gt7_rxcdrhold_i,
        gt7_rxcdrovrden_in              =>      tied_to_ground_i,
        ------------------- Receive Ports - Clock Correction Ports -----------------
--        gt7_rxclkcorcnt_out             =>      gt7_rxclkcorcnt_i,
        ------------------ Receive Ports - FPGA RX interface Ports -----------------
--        gt7_rxdata_out                  =>      gt7_rxdata_i,
        ------------------- Receive Ports - Pattern Checker Ports ------------------
        gt7_rxprbserr_out               =>      gt7_rxprbserr_i,
        gt7_rxprbssel_in                =>      gt7_rxprbssel_i,
        ------------------- Receive Ports - Pattern Checker ports ------------------
        gt7_rxprbscntreset_in           =>      gt7_rxprbscntreset_i,
        ------------------ Receive Ports - RX 8B/10B Decoder Ports -----------------
--        gt7_rxdisperr_out               =>      gt7_rxdisperr_i,
--        gt7_rxnotintable_out            =>      gt7_rxnotintable_i,
        --------------------------- Receive Ports - RX AFE -------------------------
--        gt7_gtxrxp_in                   =>      RXP_IN(7),
        ------------------------ Receive Ports - RX AFE Ports ----------------------
--        gt7_gtxrxn_in                   =>      RXN_IN(7),
        ------------------- Receive Ports - RX Buffer Bypass Ports -----------------
        gt7_rxbufreset_in               =>      gt7_rxbufreset_i,
        gt7_rxbufstatus_out             =>      gt7_rxbufstatus_i,
        -------------- Receive Ports - RX Byte and Word Alignment Ports ------------
--        gt7_rxbyteisaligned_out         =>      gt7_rxbyteisaligned_i,
--        gt7_rxbyterealign_out           =>      gt7_rxbyterealign_i,
--        gt7_rxcommadet_out              =>      gt7_rxcommadet_i,
--        gt7_rxmcommaalignen_in          =>      gt7_rxmcommaalignen_i,
--        gt7_rxpcommaalignen_in          =>      gt7_rxpcommaalignen_i,
        --------------------- Receive Ports - RX Equalizer Ports -------------------
--        gt7_rxdfelpmreset_in            =>      tied_to_ground_i,
        gt7_rxmonitorout_out            =>      gt7_rxmonitorout_i,
        gt7_rxmonitorsel_in             =>      "00",
        --------------- Receive Ports - RX Fabric Output Control Ports -------------
--        gt7_rxoutclkfabric_out          =>      gt7_rxoutclkfabric_i,
        ------------- Receive Ports - RX Initialization and Reset Ports ------------
        gt7_gtrxreset_in                =>      tied_to_ground_i,
        gt7_rxpcsreset_in               =>      tied_to_ground_i,
--        gt7_rxpmareset_in               =>      gt7_rxpmareset_i,
        ------------------ Receive Ports - RX Margin Analysis ports ----------------
--        gt7_rxlpmen_in                  =>      gt7_rxlpmen_i,
        ----------------- Receive Ports - RX Polarity Control Ports ----------------
--        gt7_rxpolarity_in               =>      gt7_rxpolarity_i,
        ----------------- Receive Ports - RX8B/10B Decoder Ports -----------------
--        gt7_rxchariscomma_out           =>      gt7_rxchariscomma_i,
--        gt7_rxcharisk_out               =>      gt7_rxcharisk_i,
        -------------- Receive Ports -RX Initialization and Reset Ports ------------
--        gt7_rxresetdone_out             =>      gt7_rxresetdone_i,
        ------------------------ TX Configurable Driver Ports ----------------------
        gt7_txpostcursor_in             =>      gt7_txpostcursor_i,
        gt7_txprecursor_in              =>      gt7_txprecursor_i,
        --------------------- TX Initialization and Reset Ports --------------------
        gt7_gttxreset_in                =>      tied_to_ground_i,
        gt7_txuserrdy_in                =>      tied_to_vcc_i,
        ---------------- Transmit Ports - 8b10b Encoder Control Ports --------------
        gt7_txchardispmode_in           =>      gt7_txchardispmode_i,
        gt7_txchardispval_in            =>      gt7_txchardispval_i,
        ------------------ Transmit Ports - Pattern Generator Ports ----------------
        gt7_txprbsforceerr_in           =>      gt7_txprbsforceerr_i,
        ---------------------- Transmit Ports - TX Buffer Ports --------------------
        gt7_txbufstatus_out             =>      gt7_txbufstatus_i,
        --------------- Transmit Ports - TX Configurable Driver Ports --------------
        gt7_txdiffctrl_in               =>      gt7_txdiffctrl_i,
        gt7_txmaincursor_in             =>      "0000000",
        ------------------ Transmit Ports - TX Data Path interface -----------------
        gt7_txdata_in                   =>      gt7_txdata_i,
        ---------------- Transmit Ports - TX Driver and OOB signaling --------------
        gt7_gtxtxn_out                  =>      TXN_OUT(7),
        gt7_gtxtxp_out                  =>      TXP_OUT(7),
        ----------- Transmit Ports - TX Fabric Clock Output Control Ports ----------
        gt7_txoutclkfabric_out          =>      gt7_txoutclkfabric_i,
        gt7_txoutclkpcs_out             =>      gt7_txoutclkpcs_i,
        --------------------- Transmit Ports - TX Gearbox Ports --------------------
        gt7_txcharisk_in                =>      gt7_txcharisk_i,
        ------------- Transmit Ports - TX Initialization and Reset Ports -----------
        gt7_txpcsreset_in               =>      tied_to_ground_i,
        gt7_txpmareset_in               =>      tied_to_ground_i,
        gt7_txresetdone_out             =>      gt7_txresetdone_i,
        ----------------- Transmit Ports - TX Polarity Control Ports ---------------
        gt7_txpolarity_in               =>      gt7_txpolarity_i,
        ------------------ Transmit Ports - pattern Generator Ports ----------------
        gt7_txprbssel_in                =>      gt7_txprbssel_i,



    --____________________________COMMON PORTS________________________________
     GT0_QPLLOUTCLK_OUT  => open,
     GT0_QPLLOUTREFCLK_OUT => open,
    --____________________________COMMON PORTS________________________________
     GT1_QPLLOUTCLK_OUT  => open,
     GT1_QPLLOUTREFCLK_OUT => open,
         sysclk_in => ipb_clk
    );





gt0_rxprbscntreset_i                         <= tied_to_ground_i;
gt0_rxprbssel_i                              <= (others => '0');
--gt0_loopback_i                               <= (others => '0');
 
gt0_txdiffctrl_i                             <= (others => '0');
gt0_rxbufreset_i                             <= tied_to_ground_i;
gt0_rxcdrhold_i                              <= tied_to_ground_i;
gt0_rxpmareset_i                             <= tied_to_ground_i;
gt0_rxpolarity_i                             <= tied_to_ground_i;
gt0_rxpd_i                                   <= (others => '0');
gt0_txprecursor_i                            <= (others => '0');
gt0_txpostcursor_i                           <= (others => '0');
gt0_txchardispmode_i                         <= (others => '0');
gt0_txchardispval_i                          <= (others => '0');
gt0_txpolarity_i                             <= '1';
gt0_txpd_i                                   <= (others => '0');
gt0_txprbsforceerr_i                         <= tied_to_ground_i;
gt0_txprbssel_i                              <= (others => '0');
------------------------------------------------------
    gt0_rxlpmen_i                                <= tied_to_vcc_i;
    
    

gt1_rxprbscntreset_i                         <= tied_to_ground_i;
gt1_rxprbssel_i                              <= (others => '0');
gt1_txdiffctrl_i                             <= (others => '0');
gt1_rxbufreset_i                             <= tied_to_ground_i;
gt1_rxcdrhold_i                              <= tied_to_ground_i;
gt1_rxpmareset_i                             <= tied_to_ground_i;
gt1_rxpolarity_i                             <= tied_to_ground_i;
gt1_rxpd_i                                   <= (others => '0');
gt1_txprecursor_i                            <= (others => '0');
gt1_txpostcursor_i                           <= (others => '0');
gt1_txchardispmode_i                         <= (others => '0');
gt1_txchardispval_i                          <= (others => '0');
gt1_txpolarity_i                             <= '1';
gt1_txpd_i                                   <= (others => '0');
gt1_txprbsforceerr_i                         <= tied_to_ground_i;
gt1_txprbssel_i                              <= (others => '0');
gt1_rxlpmen_i                                <= tied_to_vcc_i;



gt2_rxprbscntreset_i                         <= tied_to_ground_i;
gt2_rxprbssel_i                              <= (others => '0');
gt2_txdiffctrl_i                             <= (others => '0');
gt2_rxbufreset_i                             <= tied_to_ground_i;
gt2_rxcdrhold_i                              <= tied_to_ground_i;
gt2_rxpmareset_i                             <= tied_to_ground_i;
gt2_rxpolarity_i                             <= tied_to_ground_i;
gt2_rxpd_i                                   <= (others => '0');
gt2_txprecursor_i                            <= (others => '0');
gt2_txpostcursor_i                           <= (others => '0');
gt2_txchardispmode_i                         <= (others => '0');
gt2_txchardispval_i                          <= (others => '0');
gt2_txpolarity_i                             <= '1';
gt2_txpd_i                                   <= (others => '0');
gt2_txprbsforceerr_i                         <= tied_to_ground_i;
gt2_txprbssel_i                              <= (others => '0');
gt2_rxlpmen_i                                <= tied_to_vcc_i;



gt3_rxprbscntreset_i                         <= tied_to_ground_i;
gt3_rxprbssel_i                              <= (others => '0');
gt3_txdiffctrl_i                             <= (others => '0');
gt3_rxbufreset_i                             <= tied_to_ground_i;
gt3_rxcdrhold_i                              <= tied_to_ground_i;
gt3_rxpmareset_i                             <= tied_to_ground_i;
gt3_rxpolarity_i                             <= tied_to_ground_i;
gt3_rxpd_i                                   <= (others => '0');
gt3_txprecursor_i                            <= (others => '0');
gt3_txpostcursor_i                           <= (others => '0');
gt3_txchardispmode_i                         <= (others => '0');
gt3_txchardispval_i                          <= (others => '0');
gt3_txpolarity_i                             <= tied_to_ground_i;
gt3_txpd_i                                   <= (others => '0');
gt3_txprbsforceerr_i                         <= tied_to_ground_i;
gt3_txprbssel_i                              <= (others => '0');
gt3_rxlpmen_i                                <= tied_to_vcc_i;



gt4_rxprbscntreset_i                         <= tied_to_ground_i;
gt4_rxprbssel_i                              <= (others => '0');
gt4_txdiffctrl_i                             <= (others => '0');
gt4_rxbufreset_i                             <= tied_to_ground_i;
gt4_rxcdrhold_i                              <= tied_to_ground_i;
gt4_rxpmareset_i                             <= tied_to_ground_i;
gt4_rxpolarity_i                             <= tied_to_ground_i;
gt4_rxpd_i                                   <= (others => '0');
gt4_txprecursor_i                            <= (others => '0');
gt4_txpostcursor_i                           <= (others => '0');
gt4_txchardispmode_i                         <= (others => '0');
gt4_txchardispval_i                          <= (others => '0');
gt4_txpolarity_i                             <= '1';
gt4_txpd_i                                   <= (others => '0');
gt4_txprbsforceerr_i                         <= tied_to_ground_i;
gt4_txprbssel_i                              <= (others => '0');
gt4_rxlpmen_i                                <= tied_to_vcc_i;




gt5_rxprbscntreset_i                         <= tied_to_ground_i;
gt5_rxprbssel_i                              <= (others => '0');
gt5_txdiffctrl_i                             <= (others => '0');
gt5_rxbufreset_i                             <= tied_to_ground_i;
gt5_rxcdrhold_i                              <= tied_to_ground_i;
gt5_rxpmareset_i                             <= tied_to_ground_i;
gt5_rxpolarity_i                             <= tied_to_ground_i;
gt5_rxpd_i                                   <= (others => '0');
gt5_txprecursor_i                            <= (others => '0');
gt5_txpostcursor_i                           <= (others => '0');
gt5_txchardispmode_i                         <= (others => '0');
gt5_txchardispval_i                          <= (others => '0');
gt5_txpolarity_i                             <= '1';
gt5_txpd_i                                   <= (others => '0');
gt5_txprbsforceerr_i                         <= tied_to_ground_i;
gt5_txprbssel_i                              <= (others => '0');
gt5_rxlpmen_i                                <= tied_to_vcc_i;



gt6_rxprbscntreset_i                         <= tied_to_ground_i;
gt6_rxprbssel_i                              <= (others => '0');
gt6_txdiffctrl_i                             <= (others => '0');
gt6_rxbufreset_i                             <= tied_to_ground_i;
gt6_rxcdrhold_i                              <= tied_to_ground_i;
gt6_rxpmareset_i                             <= tied_to_ground_i;
gt6_rxpolarity_i                             <= tied_to_ground_i;
gt6_rxpd_i                                   <= (others => '0');
gt6_txprecursor_i                            <= (others => '0');
gt6_txpostcursor_i                           <= (others => '0');
gt6_txchardispmode_i                         <= (others => '0');
gt6_txchardispval_i                          <= (others => '0');
gt6_txpolarity_i                             <= '1';
gt6_txpd_i                                   <= (others => '0');
gt6_txprbsforceerr_i                         <= tied_to_ground_i;
gt6_txprbssel_i                              <= (others => '0');
gt6_rxlpmen_i                                <= tied_to_vcc_i;



gt7_rxprbscntreset_i                         <= tied_to_ground_i;
gt7_rxprbssel_i                              <= (others => '0');
gt7_txdiffctrl_i                             <= (others => '0');
gt7_rxbufreset_i                             <= tied_to_ground_i;
gt7_rxcdrhold_i                              <= tied_to_ground_i;
gt7_rxpmareset_i                             <= tied_to_ground_i;
gt7_rxpolarity_i                             <= tied_to_ground_i;
gt7_rxpd_i                                   <= (others => '0');
gt7_txprecursor_i                            <= (others => '0');
gt7_txpostcursor_i                           <= (others => '0');
gt7_txchardispmode_i                         <= (others => '0');
gt7_txchardispval_i                          <= (others => '0');
gt7_txpolarity_i                             <= '1';
gt7_txpd_i                                   <= (others => '0');
gt7_txprbsforceerr_i                         <= tied_to_ground_i;
gt7_txprbssel_i                              <= (others => '0');
gt7_rxlpmen_i                                <= tied_to_vcc_i;

--gt0_txcharisk_i <= '1' & fifo_out(26 downto 24);
--gt0_txcharisk_i <= "1011";
gt0_drpaddr_i <= (others => '0');
gt0_drpdi_i <= (others => '0');
gt0_drpen_i <= '0';
gt0_drpwe_i <= '0';

gt1_drpaddr_i <= (others => '0');
gt1_drpdi_i <= (others => '0');
gt1_drpen_i <= '0';
gt1_drpwe_i <= '0';

gt2_drpaddr_i <= (others => '0');
gt2_drpdi_i <= (others => '0');
gt2_drpen_i <= '0';
gt2_drpwe_i <= '0';

gt3_drpaddr_i <= (others => '0');
gt3_drpdi_i <= (others => '0');
gt3_drpen_i <= '0';
gt3_drpwe_i <= '0';

gt4_drpaddr_i <= (others => '0');
gt4_drpdi_i <= (others => '0');
gt4_drpen_i <= '0';
gt4_drpwe_i <= '0';

gt5_drpaddr_i <= (others => '0');
gt5_drpdi_i <= (others => '0');
gt5_drpen_i <= '0';
gt5_drpwe_i <= '0';

gt6_drpaddr_i <= (others => '0');
gt6_drpdi_i <= (others => '0');
gt6_drpen_i <= '0';
gt6_drpwe_i <= '0';
--
gt7_drpaddr_i <= (others => '0');
gt7_drpdi_i <= (others => '0');
gt7_drpen_i <= '0';
gt7_drpwe_i <= '0';



gt0_calibration : process (gt0_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt0_track_data_i <= '0';
	elsif rising_edge(gt0_rxusrclk_i) then
		if (gt0_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt0_track_data_i <= '1';
		end if;
	end if;
end process gt0_calibration;

gt1_calibration : process (gt1_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt1_track_data_i <= '0';
	elsif rising_edge(gt1_rxusrclk_i) then
		if (gt1_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt1_track_data_i <= '1';
		end if;
	end if;
end process gt1_calibration;

gt2_calibration : process (gt2_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt2_track_data_i <= '0';
	elsif rising_edge(gt2_rxusrclk_i) then
		if (gt2_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt2_track_data_i <= '1';
		end if;
	end if;
end process gt2_calibration;

gt3_calibration : process (gt3_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt3_track_data_i <= '0';
	elsif rising_edge(gt3_rxusrclk_i) then
		if (gt3_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt3_track_data_i <= '1';
		end if;
	end if;
end process gt3_calibration;

gt4_calibration : process (gt4_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt4_track_data_i <= '0';
	elsif rising_edge(gt4_rxusrclk_i) then
		if (gt4_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt4_track_data_i <= '1';
		end if;
	end if;
end process gt4_calibration;

gt5_calibration : process (gt5_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt5_track_data_i <= '0';
	elsif rising_edge(gt5_rxusrclk_i) then
		if (gt5_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt5_track_data_i <= '1';
		end if;
	end if;
end process gt5_calibration;

gt6_calibration : process (gt6_rxusrclk_i, soft_rst) is
begin
	if soft_rst = '1' then
		gt6_track_data_i <= '0';
	elsif rising_edge(gt6_rxusrclk_i) then
		if (gt6_rxdata_i = X"bcbc00bc" and calibrate_firefly='1') then
			gt6_track_data_i <= '1';
		end if;
	end if;
end process gt6_calibration;



--      Ethernet MAC core and PHY interface
-- In this version, consists of hard MAC core and GMII interface to external PHY
-- Can be replaced by any other MAC / PHY combination

        eth: entity work.eth_7s_rgmii
                port map(
						--refclkready => refclkready,
						clk125 => clk125,
						clk125_90 => clk125_90,
						clk200 => clk200,
						rst => rst_125,
						rgmii_txd => rgmii_txd,
						rgmii_tx_ctl => rgmii_tx_ctl,
						rgmii_txc => rgmii_txc,
						rgmii_rxd => rgmii_rxd,
						rgmii_rx_ctl => rgmii_rx_ctl,
						rgmii_rxc => rgmii_rxc,
						tx_data => mac_tx_data,
						tx_valid => mac_tx_valid,
						tx_last => mac_tx_last,
						tx_error => mac_tx_error,
						tx_ready => mac_tx_ready,
						rx_data => mac_rx_data,
						rx_valid => mac_rx_valid,
						rx_last => mac_rx_last,
						rx_error => mac_rx_error
                );

        phy_rstb <= not rst_125;

-- ipbus control logic

        ipbus: entity work.ipbus_ctrl
                port map(
                        mac_clk => clk125,
                        rst_macclk => rst_125,
                        ipb_clk => ipb_clk,
                        rst_ipb => rst_ipb,
                        mac_rx_data => mac_rx_data,
                        mac_rx_valid => mac_rx_valid,
                        mac_rx_last => mac_rx_last,
                        mac_rx_error => mac_rx_error,
                        mac_tx_data => mac_tx_data,
                        mac_tx_valid => mac_tx_valid,
                        mac_tx_last => mac_tx_last,
                        mac_tx_error => mac_tx_error,
                        mac_tx_ready => mac_tx_ready,
                        ipb_out => ipb_master_out,
                        ipb_in => ipb_master_in,
                        mac_addr => mac_addr,
                        ip_addr => ip_addr
--                        pkt_rx => pkt_rx,
--                        pkt_tx => pkt_tx,
--                        pkt_rx_led => pkt_rx_led,
--                        pkt_tx_led => pkt_tx_led
                );

	mac_addr <= X"020ddba11550"; -- Careful here, arbitrary addresses do not always work
	ip_addr <= X"c0a8c855"; -- 192.168.200.85
	--mac_addr <= X"0090b81e44f9"; -- Careful here, arbitrary addresses do not always work
    --ip_addr <= X"0a2071da"; -- 10.32.113.218
    
	ILA0: ENTITY work.ila_0
		port map (
			probe5 => gt3_txcharisk_i & X"0000000",
			probe6 => "000000000",
			clk => gt0_txusrclk_i,
			probe0 => gt3_txdata_i,
			probe1(0) => '0',
			probe2(0) => '0',
			probe3 => (others => '0'),
			probe4(0) => '0'
		);
--	ILA1: ENTITY work.ila_0
--		port map (
--			probe5 => fifo_out,
----			probe6 => (others => '0'),
----			probe6(7) => gt0_rxcommadet_i,
----			probe6(6) => gt0_rxbyteisaligned_i,
--			probe6 => track_data_i & "00000000",
----			probe6(4) => data_valid,
----			probe6(3 downto 0) => charisk,
--			clk => gt0_rxusrclk_i,
--			probe0 => fifo_in,
--			probe1(0) => fifo_wr,
--			probe2(0) => '0',
--			probe3 => fifo_buf,
--			probe4(0) => rxfsmresetdone_i
--		);
--	vio0: entity work.vio_0
--		
--			port map (
--				probe_out0(0) => fifo_rd,
--				clk => ipb_clk
--		);


        slaves: entity work.ipbus_mu3e port map(
                ipb_clk => ipb_clk,
                ipb_rst => slaves_rst,
                ipb_in => ipb_master_out,
                ipb_out => ipb_master_in,
                nuke => nuke,
                --nuke => sys_rst
                soft_rst => soft_rst,
                clk_rd => gt0_txusrclk_i,
                clk_wr => gt0_rxusrclk_i,
                fifo_DO => fifo_out,
                --in_cmd => ipbus_data_out
                --fifo_rd => fifo_rd
                fifo_wr => fifo_wr,
                fifo_DI => fifo_buf,
                charisk => txcharisk_i,
                calibrate => calibrate_firefly,
                data_valid => data_valid_i,
                mgt_adrs => mgt_adrs,
                mgt_mask => mgt_mask
                --bytecount => bytecount
        );
        slaves_rst <= rst_ipb or soft_rst;
        
        fifo_buffer_mon : process (gt0_rxusrclk_i, rst_ipb) is
        begin
        	if rst_ipb = '1' then
        		fifo_buf <= (others => '0');
        		fifo_wr <= '0';
        	elsif rising_edge(gt0_rxusrclk_i) then
        		if (fifo_buf /= fifo_in) and (rxfsmresetdone_i = '1') then--and (fourcommas = '0') then
        			fifo_wr <= '1';
        		else
        			fifo_wr <= '0';
        		end if;
        		fifo_buf <= fifo_in;
        		
        	end if;
        end process fifo_buffer_mon;
        --fifo_in <= fifo_buf;
        --fifo_rd <= '1';
        
        --fourcommas <= '1' when fifo_in = X"bcbcbcbc" else '0';
  		gt0_txcharisk_i <= txcharisk_i when mgt_mask(0)='1' else X"f";
		gt1_txcharisk_i <= txcharisk_i when mgt_mask(1)='1' else X"f";
		gt2_txcharisk_i <= txcharisk_i when mgt_mask(2)='1' else X"f";
		gt3_txcharisk_i <= txcharisk_i when mgt_mask(3)='1' else X"f";
		gt4_txcharisk_i <= txcharisk_i when mgt_mask(4)='1' else X"f";
		gt5_txcharisk_i <= txcharisk_i when mgt_mask(5)='1' else X"f";
		gt6_txcharisk_i <= txcharisk_i when mgt_mask(6)='1' else X"f";
		gt7_txcharisk_i <= txcharisk_i when mgt_mask(7)='1' else X"f"; 
 
		data_valid_i <= gt0_track_data_i when mgt_adrs="000" else
						gt1_track_data_i when mgt_adrs="001" else
						gt2_track_data_i when mgt_adrs="010" else
						gt3_track_data_i when mgt_adrs="011" else
						gt4_track_data_i when mgt_adrs="100" else
						gt5_track_data_i when mgt_adrs="101" else
						gt6_track_data_i when mgt_adrs="110" ;
--						gt7_track_data_i when mgt_adrs="111";
        
		rxfsmresetdone_i <= gt0_rxfsmresetdone_i when mgt_adrs="000" else
							gt1_rxfsmresetdone_i when mgt_adrs="001" else
							gt2_rxfsmresetdone_i when mgt_adrs="010" else
							gt3_rxfsmresetdone_i when mgt_adrs="011" else
							gt4_rxfsmresetdone_i when mgt_adrs="100" else
							gt5_rxfsmresetdone_i when mgt_adrs="101" else
							gt6_rxfsmresetdone_i when mgt_adrs="110" ;
--							gt7_rxfsmresetdone_i when mgt_adrs="111";
					
					
					
 		gt0_txdata_i <= fifo_out when mgt_mask(0)='1' else X"bcbcbcbc";
		gt1_txdata_i <= fifo_out when mgt_mask(1)='1' else X"bcbcbcbc";
		gt2_txdata_i <= fifo_out when mgt_mask(2)='1' else X"bcbcbcbc";
		gt3_txdata_i <= fifo_out when mgt_mask(3)='1' else X"bcbcbcbc";
		gt4_txdata_i <= fifo_out when mgt_mask(4)='1' else X"bcbcbcbc";
		gt5_txdata_i <= fifo_out when mgt_mask(5)='1' else X"bcbcbcbc";
		gt6_txdata_i <= fifo_out when mgt_mask(6)='1' else X"bcbcbcbc";
		gt7_txdata_i <= fifo_out when mgt_mask(7)='1' else X"bcbcbcbc";
		

--		
					

led_ipbus(0) <= '1';-- when (ipbus_data_in = X"abcdef12") else
			-- '0';
led_ipbus(1) <= '1';-- when (ipbus_data_in = X"12345678") else
			-- '0';
			
  CLK_OE_L  		<= '0';
  CLK_RST_L 		<= not soft_rst;
  TX_MGT_RST_L		<= '1';
  TX_MGT_SLCT_L 	<= '1';
  TX_CLK_RST_L		<= '1';
  TX_CLK_SLCT_L 	<= '1';
			 
end rtl;