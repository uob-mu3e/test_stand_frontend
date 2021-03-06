/* Configuration header for STiC3 ASIC. */
/* Input file: mutrig2_DCR */
/* Config name:mutrig2_DCR.h */


#define STIC3_CONFIG_LEN_BITS 2719
#define STIC3_CONFIG_LEN_BYTES 340


//-----
//--- Parameter dump of configuration mutrig2_DCR.h ---
//-----
/*
O_GEN_IDLE_SIGNAL = 1
SYNC_CH_RST = 1
EXT_TRIG_MODE = 0
EXT_TRIG_SIGN_FORW_TIME = 0
EXT_TRIG_BACK_TIME = 0
EXT_TRIG_FORW_TIME = 0
MS_LIMITS = 0
MS_SWITCH_SEL = 0
MS_DEBUG = 0
PRBS_DEBUG = 0
PRBS_SINGLE = 0
FAST_TRANS_MODE = 0
PLL1_SETCOARSE = 0
PLL1_ENVCOMONITOR = 0
DISABLE_COARSE = 0
PLL_LOL_dbg = 0
EN_CH_EVT_CNT = 0
ANODE_FLAG_CH0 = 1
CATHODE_FLAG_CH0 = 1
S_SWITCH_CH0 = 1
SORD_CH0 = 0
SORD_NOT_CH0 = 1
EDGE_CH0 = 1
EDGE_CML_CH0 = 0
DAC_CMLSCALE_CH0 = 0
DMON_ENA_CH0 = 0
DMON_SW_CH0 = 0
TDCTEST_CH0 = 1
AMON_CTRL_CH0 = 0
COMP_SPI_CH0 = 0
DAC_SIPM_SC_CH0 = 0
DAC_SIPM_CH0 = 23
DAC_TTHRESH_SC_CH0 = 0
DAC_TTHRESH_CH0 = 20
DAC_AMPCOM_SC_CH0 = 2
DAC_AMPCOM_CH0 = 0
DAC_INPUTBIAS_SC_CH0 = 0
DAC_INPUTBIAS_CH0 = 4
DAC_ETHRESH_CH0 = 0
DAC_EBIAS_CH0 = 0
DAC_POLE_SC_CH0 = 0
DAC_POLE_CH0 = 3f
DAC_CML_CH0 = 8
DAC_DELAY_CH0 = 0
DAC_DELAY_BIT1_CH0 = 0
DAC_CHANNEL_MASK_CH0 = 0
RECV_ALL_CH0 = 1
ANODE_FLAG_CH1 = 1
CATHODE_FLAG_CH1 = 1
S_SWITCH_CH1 = 1
SORD_CH1 = 0
SORD_NOT_CH1 = 1
EDGE_CH1 = 1
EDGE_CML_CH1 = 0
DAC_CMLSCALE_CH1 = 0
DMON_ENA_CH1 = 0
DMON_SW_CH1 = 0
TDCTEST_CH1 = 1
AMON_CTRL_CH1 = 0
COMP_SPI_CH1 = 0
DAC_SIPM_SC_CH1 = 0
DAC_SIPM_CH1 = f
DAC_TTHRESH_SC_CH1 = 0
DAC_TTHRESH_CH1 = 20
DAC_AMPCOM_SC_CH1 = 2
DAC_AMPCOM_CH1 = 0
DAC_INPUTBIAS_SC_CH1 = 0
DAC_INPUTBIAS_CH1 = 4
DAC_ETHRESH_CH1 = 0
DAC_EBIAS_CH1 = 0
DAC_POLE_SC_CH1 = 0
DAC_POLE_CH1 = 3f
DAC_CML_CH1 = 8
DAC_DELAY_CH1 = 0
DAC_DELAY_BIT1_CH1 = 0
DAC_CHANNEL_MASK_CH1 = 0
RECV_ALL_CH1 = 1
ANODE_FLAG_CH2 = 1
CATHODE_FLAG_CH2 = 1
S_SWITCH_CH2 = 1
SORD_CH2 = 0
SORD_NOT_CH2 = 1
EDGE_CH2 = 1
EDGE_CML_CH2 = 0
DAC_CMLSCALE_CH2 = 0
DMON_ENA_CH2 = 0
DMON_SW_CH2 = 0
TDCTEST_CH2 = 1
AMON_CTRL_CH2 = 0
COMP_SPI_CH2 = 0
DAC_SIPM_SC_CH2 = 0
DAC_SIPM_CH2 = f
DAC_TTHRESH_SC_CH2 = 0
DAC_TTHRESH_CH2 = 20
DAC_AMPCOM_SC_CH2 = 2
DAC_AMPCOM_CH2 = 0
DAC_INPUTBIAS_SC_CH2 = 0
DAC_INPUTBIAS_CH2 = 4
DAC_ETHRESH_CH2 = 0
DAC_EBIAS_CH2 = 0
DAC_POLE_SC_CH2 = 0
DAC_POLE_CH2 = 3f
DAC_CML_CH2 = 8
DAC_DELAY_CH2 = 0
DAC_DELAY_BIT1_CH2 = 0
DAC_CHANNEL_MASK_CH2 = 0
RECV_ALL_CH2 = 1
ANODE_FLAG_CH3 = 1
CATHODE_FLAG_CH3 = 1
S_SWITCH_CH3 = 1
SORD_CH3 = 0
SORD_NOT_CH3 = 1
EDGE_CH3 = 1
EDGE_CML_CH3 = 0
DAC_CMLSCALE_CH3 = 0
DMON_ENA_CH3 = 0
DMON_SW_CH3 = 0
TDCTEST_CH3 = 1
AMON_CTRL_CH3 = 0
COMP_SPI_CH3 = 0
DAC_SIPM_SC_CH3 = 0
DAC_SIPM_CH3 = f
DAC_TTHRESH_SC_CH3 = 0
DAC_TTHRESH_CH3 = 20
DAC_AMPCOM_SC_CH3 = 2
DAC_AMPCOM_CH3 = 0
DAC_INPUTBIAS_SC_CH3 = 0
DAC_INPUTBIAS_CH3 = 4
DAC_ETHRESH_CH3 = 0
DAC_EBIAS_CH3 = 0
DAC_POLE_SC_CH3 = 0
DAC_POLE_CH3 = 3f
DAC_CML_CH3 = 8
DAC_DELAY_CH3 = 0
DAC_DELAY_BIT1_CH3 = 0
DAC_CHANNEL_MASK_CH3 = 0
RECV_ALL_CH3 = 1
ANODE_FLAG_CH4 = 1
CATHODE_FLAG_CH4 = 1
S_SWITCH_CH4 = 1
SORD_CH4 = 0
SORD_NOT_CH4 = 1
EDGE_CH4 = 1
EDGE_CML_CH4 = 0
DAC_CMLSCALE_CH4 = 0
DMON_ENA_CH4 = 0
DMON_SW_CH4 = 0
TDCTEST_CH4 = 1
AMON_CTRL_CH4 = 0
COMP_SPI_CH4 = 0
DAC_SIPM_SC_CH4 = 0
DAC_SIPM_CH4 = f
DAC_TTHRESH_SC_CH4 = 0
DAC_TTHRESH_CH4 = 20
DAC_AMPCOM_SC_CH4 = 2
DAC_AMPCOM_CH4 = 0
DAC_INPUTBIAS_SC_CH4 = 0
DAC_INPUTBIAS_CH4 = 4
DAC_ETHRESH_CH4 = 0
DAC_EBIAS_CH4 = 0
DAC_POLE_SC_CH4 = 0
DAC_POLE_CH4 = 3f
DAC_CML_CH4 = 8
DAC_DELAY_CH4 = 0
DAC_DELAY_BIT1_CH4 = 0
DAC_CHANNEL_MASK_CH4 = 0
RECV_ALL_CH4 = 1
ANODE_FLAG_CH5 = 1
CATHODE_FLAG_CH5 = 1
S_SWITCH_CH5 = 1
SORD_CH5 = 0
SORD_NOT_CH5 = 1
EDGE_CH5 = 1
EDGE_CML_CH5 = 0
DAC_CMLSCALE_CH5 = 0
DMON_ENA_CH5 = 0
DMON_SW_CH5 = 0
TDCTEST_CH5 = 1
AMON_CTRL_CH5 = 0
COMP_SPI_CH5 = 0
DAC_SIPM_SC_CH5 = 0
DAC_SIPM_CH5 = f
DAC_TTHRESH_SC_CH5 = 0
DAC_TTHRESH_CH5 = 20
DAC_AMPCOM_SC_CH5 = 2
DAC_AMPCOM_CH5 = 0
DAC_INPUTBIAS_SC_CH5 = 0
DAC_INPUTBIAS_CH5 = 4
DAC_ETHRESH_CH5 = 0
DAC_EBIAS_CH5 = 0
DAC_POLE_SC_CH5 = 0
DAC_POLE_CH5 = 3f
DAC_CML_CH5 = 8
DAC_DELAY_CH5 = 0
DAC_DELAY_BIT1_CH5 = 0
DAC_CHANNEL_MASK_CH5 = 0
RECV_ALL_CH5 = 1
ANODE_FLAG_CH6 = 1
CATHODE_FLAG_CH6 = 1
S_SWITCH_CH6 = 1
SORD_CH6 = 0
SORD_NOT_CH6 = 1
EDGE_CH6 = 1
EDGE_CML_CH6 = 0
DAC_CMLSCALE_CH6 = 0
DMON_ENA_CH6 = 0
DMON_SW_CH6 = 0
TDCTEST_CH6 = 1
AMON_CTRL_CH6 = 0
COMP_SPI_CH6 = 0
DAC_SIPM_SC_CH6 = 0
DAC_SIPM_CH6 = f
DAC_TTHRESH_SC_CH6 = 0
DAC_TTHRESH_CH6 = 20
DAC_AMPCOM_SC_CH6 = 2
DAC_AMPCOM_CH6 = 0
DAC_INPUTBIAS_SC_CH6 = 0
DAC_INPUTBIAS_CH6 = 4
DAC_ETHRESH_CH6 = 0
DAC_EBIAS_CH6 = 0
DAC_POLE_SC_CH6 = 0
DAC_POLE_CH6 = 3f
DAC_CML_CH6 = 8
DAC_DELAY_CH6 = 0
DAC_DELAY_BIT1_CH6 = 0
DAC_CHANNEL_MASK_CH6 = 0
RECV_ALL_CH6 = 1
ANODE_FLAG_CH7 = 1
CATHODE_FLAG_CH7 = 1
S_SWITCH_CH7 = 1
SORD_CH7 = 0
SORD_NOT_CH7 = 1
EDGE_CH7 = 1
EDGE_CML_CH7 = 0
DAC_CMLSCALE_CH7 = 0
DMON_ENA_CH7 = 0
DMON_SW_CH7 = 0
TDCTEST_CH7 = 1
AMON_CTRL_CH7 = 0
COMP_SPI_CH7 = 0
DAC_SIPM_SC_CH7 = 0
DAC_SIPM_CH7 = f
DAC_TTHRESH_SC_CH7 = 0
DAC_TTHRESH_CH7 = 20
DAC_AMPCOM_SC_CH7 = 2
DAC_AMPCOM_CH7 = 0
DAC_INPUTBIAS_SC_CH7 = 0
DAC_INPUTBIAS_CH7 = 4
DAC_ETHRESH_CH7 = 0
DAC_EBIAS_CH7 = 0
DAC_POLE_SC_CH7 = 0
DAC_POLE_CH7 = 3f
DAC_CML_CH7 = 8
DAC_DELAY_CH7 = 0
DAC_DELAY_BIT1_CH7 = 0
DAC_CHANNEL_MASK_CH7 = 0
RECV_ALL_CH7 = 1
ANODE_FLAG_CH8 = 1
CATHODE_FLAG_CH8 = 1
S_SWITCH_CH8 = 1
SORD_CH8 = 0
SORD_NOT_CH8 = 1
EDGE_CH8 = 1
EDGE_CML_CH8 = 0
DAC_CMLSCALE_CH8 = 0
DMON_ENA_CH8 = 0
DMON_SW_CH8 = 0
TDCTEST_CH8 = 1
AMON_CTRL_CH8 = 0
COMP_SPI_CH8 = 0
DAC_SIPM_SC_CH8 = 0
DAC_SIPM_CH8 = f
DAC_TTHRESH_SC_CH8 = 0
DAC_TTHRESH_CH8 = 20
DAC_AMPCOM_SC_CH8 = 2
DAC_AMPCOM_CH8 = 0
DAC_INPUTBIAS_SC_CH8 = 0
DAC_INPUTBIAS_CH8 = 4
DAC_ETHRESH_CH8 = 0
DAC_EBIAS_CH8 = 0
DAC_POLE_SC_CH8 = 0
DAC_POLE_CH8 = 3f
DAC_CML_CH8 = 8
DAC_DELAY_CH8 = 0
DAC_DELAY_BIT1_CH8 = 0
DAC_CHANNEL_MASK_CH8 = 0
RECV_ALL_CH8 = 1
ANODE_FLAG_CH9 = 1
CATHODE_FLAG_CH9 = 1
S_SWITCH_CH9 = 1
SORD_CH9 = 0
SORD_NOT_CH9 = 1
EDGE_CH9 = 1
EDGE_CML_CH9 = 0
DAC_CMLSCALE_CH9 = 0
DMON_ENA_CH9 = 0
DMON_SW_CH9 = 0
TDCTEST_CH9 = 1
AMON_CTRL_CH9 = 0
COMP_SPI_CH9 = 0
DAC_SIPM_SC_CH9 = 0
DAC_SIPM_CH9 = f
DAC_TTHRESH_SC_CH9 = 0
DAC_TTHRESH_CH9 = 20
DAC_AMPCOM_SC_CH9 = 2
DAC_AMPCOM_CH9 = 0
DAC_INPUTBIAS_SC_CH9 = 0
DAC_INPUTBIAS_CH9 = 4
DAC_ETHRESH_CH9 = 0
DAC_EBIAS_CH9 = 0
DAC_POLE_SC_CH9 = 0
DAC_POLE_CH9 = 3f
DAC_CML_CH9 = 8
DAC_DELAY_CH9 = 0
DAC_DELAY_BIT1_CH9 = 0
DAC_CHANNEL_MASK_CH9 = 0
RECV_ALL_CH9 = 1
ANODE_FLAG_CH10 = 1
CATHODE_FLAG_CH10 = 1
S_SWITCH_CH10 = 1
SORD_CH10 = 0
SORD_NOT_CH10 = 1
EDGE_CH10 = 1
EDGE_CML_CH10 = 0
DAC_CMLSCALE_CH10 = 0
DMON_ENA_CH10 = 0
DMON_SW_CH10 = 0
TDCTEST_CH10 = 1
AMON_CTRL_CH10 = 0
COMP_SPI_CH10 = 0
DAC_SIPM_SC_CH10 = 0
DAC_SIPM_CH10 = f
DAC_TTHRESH_SC_CH10 = 0
DAC_TTHRESH_CH10 = 20
DAC_AMPCOM_SC_CH10 = 2
DAC_AMPCOM_CH10 = 0
DAC_INPUTBIAS_SC_CH10 = 0
DAC_INPUTBIAS_CH10 = 4
DAC_ETHRESH_CH10 = 0
DAC_EBIAS_CH10 = 0
DAC_POLE_SC_CH10 = 0
DAC_POLE_CH10 = 3f
DAC_CML_CH10 = 8
DAC_DELAY_CH10 = 0
DAC_DELAY_BIT1_CH10 = 0
DAC_CHANNEL_MASK_CH10 = 0
RECV_ALL_CH10 = 1
ANODE_FLAG_CH11 = 1
CATHODE_FLAG_CH11 = 1
S_SWITCH_CH11 = 1
SORD_CH11 = 0
SORD_NOT_CH11 = 1
EDGE_CH11 = 1
EDGE_CML_CH11 = 0
DAC_CMLSCALE_CH11 = 0
DMON_ENA_CH11 = 0
DMON_SW_CH11 = 0
TDCTEST_CH11 = 1
AMON_CTRL_CH11 = 0
COMP_SPI_CH11 = 0
DAC_SIPM_SC_CH11 = 0
DAC_SIPM_CH11 = f
DAC_TTHRESH_SC_CH11 = 0
DAC_TTHRESH_CH11 = 20
DAC_AMPCOM_SC_CH11 = 2
DAC_AMPCOM_CH11 = 0
DAC_INPUTBIAS_SC_CH11 = 0
DAC_INPUTBIAS_CH11 = 4
DAC_ETHRESH_CH11 = 0
DAC_EBIAS_CH11 = 0
DAC_POLE_SC_CH11 = 0
DAC_POLE_CH11 = 3f
DAC_CML_CH11 = 8
DAC_DELAY_CH11 = 0
DAC_DELAY_BIT1_CH11 = 0
DAC_CHANNEL_MASK_CH11 = 0
RECV_ALL_CH11 = 1
ANODE_FLAG_CH12 = 1
CATHODE_FLAG_CH12 = 1
S_SWITCH_CH12 = 1
SORD_CH12 = 0
SORD_NOT_CH12 = 1
EDGE_CH12 = 1
EDGE_CML_CH12 = 0
DAC_CMLSCALE_CH12 = 0
DMON_ENA_CH12 = 0
DMON_SW_CH12 = 0
TDCTEST_CH12 = 1
AMON_CTRL_CH12 = 0
COMP_SPI_CH12 = 0
DAC_SIPM_SC_CH12 = 0
DAC_SIPM_CH12 = f
DAC_TTHRESH_SC_CH12 = 0
DAC_TTHRESH_CH12 = 20
DAC_AMPCOM_SC_CH12 = 2
DAC_AMPCOM_CH12 = 0
DAC_INPUTBIAS_SC_CH12 = 0
DAC_INPUTBIAS_CH12 = 4
DAC_ETHRESH_CH12 = 0
DAC_EBIAS_CH12 = 0
DAC_POLE_SC_CH12 = 0
DAC_POLE_CH12 = 3f
DAC_CML_CH12 = 8
DAC_DELAY_CH12 = 0
DAC_DELAY_BIT1_CH12 = 0
DAC_CHANNEL_MASK_CH12 = 0
RECV_ALL_CH12 = 1
ANODE_FLAG_CH13 = 1
CATHODE_FLAG_CH13 = 1
S_SWITCH_CH13 = 1
SORD_CH13 = 0
SORD_NOT_CH13 = 1
EDGE_CH13 = 1
EDGE_CML_CH13 = 0
DAC_CMLSCALE_CH13 = 0
DMON_ENA_CH13 = 0
DMON_SW_CH13 = 0
TDCTEST_CH13 = 1
AMON_CTRL_CH13 = 0
COMP_SPI_CH13 = 0
DAC_SIPM_SC_CH13 = 0
DAC_SIPM_CH13 = f
DAC_TTHRESH_SC_CH13 = 0
DAC_TTHRESH_CH13 = 20
DAC_AMPCOM_SC_CH13 = 2
DAC_AMPCOM_CH13 = 0
DAC_INPUTBIAS_SC_CH13 = 0
DAC_INPUTBIAS_CH13 = 4
DAC_ETHRESH_CH13 = 0
DAC_EBIAS_CH13 = 0
DAC_POLE_SC_CH13 = 0
DAC_POLE_CH13 = 3f
DAC_CML_CH13 = 8
DAC_DELAY_CH13 = 0
DAC_DELAY_BIT1_CH13 = 0
DAC_CHANNEL_MASK_CH13 = 0
RECV_ALL_CH13 = 1
ANODE_FLAG_CH14 = 1
CATHODE_FLAG_CH14 = 1
S_SWITCH_CH14 = 1
SORD_CH14 = 0
SORD_NOT_CH14 = 1
EDGE_CH14 = 1
EDGE_CML_CH14 = 0
DAC_CMLSCALE_CH14 = 0
DMON_ENA_CH14 = 0
DMON_SW_CH14 = 0
TDCTEST_CH14 = 1
AMON_CTRL_CH14 = 0
COMP_SPI_CH14 = 0
DAC_SIPM_SC_CH14 = 0
DAC_SIPM_CH14 = f
DAC_TTHRESH_SC_CH14 = 0
DAC_TTHRESH_CH14 = 20
DAC_AMPCOM_SC_CH14 = 2
DAC_AMPCOM_CH14 = 0
DAC_INPUTBIAS_SC_CH14 = 0
DAC_INPUTBIAS_CH14 = 4
DAC_ETHRESH_CH14 = 0
DAC_EBIAS_CH14 = 0
DAC_POLE_SC_CH14 = 0
DAC_POLE_CH14 = 3f
DAC_CML_CH14 = 8
DAC_DELAY_CH14 = 0
DAC_DELAY_BIT1_CH14 = 0
DAC_CHANNEL_MASK_CH14 = 0
RECV_ALL_CH14 = 1
ANODE_FLAG_CH15 = 1
CATHODE_FLAG_CH15 = 1
S_SWITCH_CH15 = 1
SORD_CH15 = 0
SORD_NOT_CH15 = 1
EDGE_CH15 = 1
EDGE_CML_CH15 = 0
DAC_CMLSCALE_CH15 = 0
DMON_ENA_CH15 = 0
DMON_SW_CH15 = 0
TDCTEST_CH15 = 1
AMON_CTRL_CH15 = 0
COMP_SPI_CH15 = 0
DAC_SIPM_SC_CH15 = 0
DAC_SIPM_CH15 = f
DAC_TTHRESH_SC_CH15 = 0
DAC_TTHRESH_CH15 = 20
DAC_AMPCOM_SC_CH15 = 2
DAC_AMPCOM_CH15 = 0
DAC_INPUTBIAS_SC_CH15 = 0
DAC_INPUTBIAS_CH15 = 4
DAC_ETHRESH_CH15 = 0
DAC_EBIAS_CH15 = 0
DAC_POLE_SC_CH15 = 0
DAC_POLE_CH15 = 3f
DAC_CML_CH15 = 8
DAC_DELAY_CH15 = 0
DAC_DELAY_BIT1_CH15 = 0
DAC_CHANNEL_MASK_CH15 = 0
RECV_ALL_CH15 = 1
ANODE_FLAG_CH16 = 1
CATHODE_FLAG_CH16 = 1
S_SWITCH_CH16 = 1
SORD_CH16 = 0
SORD_NOT_CH16 = 1
EDGE_CH16 = 1
EDGE_CML_CH16 = 0
DAC_CMLSCALE_CH16 = 0
DMON_ENA_CH16 = 0
DMON_SW_CH16 = 0
TDCTEST_CH16 = 1
AMON_CTRL_CH16 = 0
COMP_SPI_CH16 = 0
DAC_SIPM_SC_CH16 = 0
DAC_SIPM_CH16 = f
DAC_TTHRESH_SC_CH16 = 0
DAC_TTHRESH_CH16 = 20
DAC_AMPCOM_SC_CH16 = 2
DAC_AMPCOM_CH16 = 0
DAC_INPUTBIAS_SC_CH16 = 0
DAC_INPUTBIAS_CH16 = 4
DAC_ETHRESH_CH16 = 0
DAC_EBIAS_CH16 = 0
DAC_POLE_SC_CH16 = 0
DAC_POLE_CH16 = 3f
DAC_CML_CH16 = 8
DAC_DELAY_CH16 = 0
DAC_DELAY_BIT1_CH16 = 0
DAC_CHANNEL_MASK_CH16 = 0
RECV_ALL_CH16 = 1
ANODE_FLAG_CH17 = 1
CATHODE_FLAG_CH17 = 1
S_SWITCH_CH17 = 1
SORD_CH17 = 0
SORD_NOT_CH17 = 1
EDGE_CH17 = 1
EDGE_CML_CH17 = 0
DAC_CMLSCALE_CH17 = 0
DMON_ENA_CH17 = 0
DMON_SW_CH17 = 0
TDCTEST_CH17 = 1
AMON_CTRL_CH17 = 0
COMP_SPI_CH17 = 0
DAC_SIPM_SC_CH17 = 0
DAC_SIPM_CH17 = f
DAC_TTHRESH_SC_CH17 = 0
DAC_TTHRESH_CH17 = 20
DAC_AMPCOM_SC_CH17 = 2
DAC_AMPCOM_CH17 = 0
DAC_INPUTBIAS_SC_CH17 = 0
DAC_INPUTBIAS_CH17 = 4
DAC_ETHRESH_CH17 = 0
DAC_EBIAS_CH17 = 0
DAC_POLE_SC_CH17 = 0
DAC_POLE_CH17 = 3f
DAC_CML_CH17 = 8
DAC_DELAY_CH17 = 0
DAC_DELAY_BIT1_CH17 = 0
DAC_CHANNEL_MASK_CH17 = 0
RECV_ALL_CH17 = 1
ANODE_FLAG_CH18 = 1
CATHODE_FLAG_CH18 = 1
S_SWITCH_CH18 = 1
SORD_CH18 = 0
SORD_NOT_CH18 = 1
EDGE_CH18 = 1
EDGE_CML_CH18 = 0
DAC_CMLSCALE_CH18 = 0
DMON_ENA_CH18 = 0
DMON_SW_CH18 = 0
TDCTEST_CH18 = 1
AMON_CTRL_CH18 = 0
COMP_SPI_CH18 = 0
DAC_SIPM_SC_CH18 = 0
DAC_SIPM_CH18 = f
DAC_TTHRESH_SC_CH18 = 0
DAC_TTHRESH_CH18 = 20
DAC_AMPCOM_SC_CH18 = 2
DAC_AMPCOM_CH18 = 0
DAC_INPUTBIAS_SC_CH18 = 0
DAC_INPUTBIAS_CH18 = 4
DAC_ETHRESH_CH18 = 0
DAC_EBIAS_CH18 = 0
DAC_POLE_SC_CH18 = 0
DAC_POLE_CH18 = 3f
DAC_CML_CH18 = 8
DAC_DELAY_CH18 = 0
DAC_DELAY_BIT1_CH18 = 0
DAC_CHANNEL_MASK_CH18 = 0
RECV_ALL_CH18 = 1
ANODE_FLAG_CH19 = 1
CATHODE_FLAG_CH19 = 1
S_SWITCH_CH19 = 1
SORD_CH19 = 0
SORD_NOT_CH19 = 1
EDGE_CH19 = 1
EDGE_CML_CH19 = 0
DAC_CMLSCALE_CH19 = 0
DMON_ENA_CH19 = 0
DMON_SW_CH19 = 0
TDCTEST_CH19 = 1
AMON_CTRL_CH19 = 0
COMP_SPI_CH19 = 0
DAC_SIPM_SC_CH19 = 0
DAC_SIPM_CH19 = f
DAC_TTHRESH_SC_CH19 = 0
DAC_TTHRESH_CH19 = 20
DAC_AMPCOM_SC_CH19 = 2
DAC_AMPCOM_CH19 = 0
DAC_INPUTBIAS_SC_CH19 = 0
DAC_INPUTBIAS_CH19 = 4
DAC_ETHRESH_CH19 = 0
DAC_EBIAS_CH19 = 0
DAC_POLE_SC_CH19 = 0
DAC_POLE_CH19 = 3f
DAC_CML_CH19 = 8
DAC_DELAY_CH19 = 0
DAC_DELAY_BIT1_CH19 = 0
DAC_CHANNEL_MASK_CH19 = 0
RECV_ALL_CH19 = 1
ANODE_FLAG_CH20 = 1
CATHODE_FLAG_CH20 = 1
S_SWITCH_CH20 = 1
SORD_CH20 = 0
SORD_NOT_CH20 = 1
EDGE_CH20 = 1
EDGE_CML_CH20 = 0
DAC_CMLSCALE_CH20 = 0
DMON_ENA_CH20 = 0
DMON_SW_CH20 = 0
TDCTEST_CH20 = 1
AMON_CTRL_CH20 = 0
COMP_SPI_CH20 = 0
DAC_SIPM_SC_CH20 = 0
DAC_SIPM_CH20 = f
DAC_TTHRESH_SC_CH20 = 0
DAC_TTHRESH_CH20 = 20
DAC_AMPCOM_SC_CH20 = 2
DAC_AMPCOM_CH20 = 0
DAC_INPUTBIAS_SC_CH20 = 0
DAC_INPUTBIAS_CH20 = 4
DAC_ETHRESH_CH20 = 0
DAC_EBIAS_CH20 = 0
DAC_POLE_SC_CH20 = 0
DAC_POLE_CH20 = 3f
DAC_CML_CH20 = 8
DAC_DELAY_CH20 = 0
DAC_DELAY_BIT1_CH20 = 0
DAC_CHANNEL_MASK_CH20 = 0
RECV_ALL_CH20 = 1
ANODE_FLAG_CH21 = 1
CATHODE_FLAG_CH21 = 1
S_SWITCH_CH21 = 1
SORD_CH21 = 0
SORD_NOT_CH21 = 1
EDGE_CH21 = 1
EDGE_CML_CH21 = 0
DAC_CMLSCALE_CH21 = 0
DMON_ENA_CH21 = 0
DMON_SW_CH21 = 0
TDCTEST_CH21 = 1
AMON_CTRL_CH21 = 0
COMP_SPI_CH21 = 0
DAC_SIPM_SC_CH21 = 0
DAC_SIPM_CH21 = f
DAC_TTHRESH_SC_CH21 = 0
DAC_TTHRESH_CH21 = 20
DAC_AMPCOM_SC_CH21 = 2
DAC_AMPCOM_CH21 = 0
DAC_INPUTBIAS_SC_CH21 = 0
DAC_INPUTBIAS_CH21 = 4
DAC_ETHRESH_CH21 = 0
DAC_EBIAS_CH21 = 0
DAC_POLE_SC_CH21 = 0
DAC_POLE_CH21 = 3f
DAC_CML_CH21 = 8
DAC_DELAY_CH21 = 0
DAC_DELAY_BIT1_CH21 = 0
DAC_CHANNEL_MASK_CH21 = 0
RECV_ALL_CH21 = 1
ANODE_FLAG_CH22 = 1
CATHODE_FLAG_CH22 = 1
S_SWITCH_CH22 = 1
SORD_CH22 = 0
SORD_NOT_CH22 = 1
EDGE_CH22 = 1
EDGE_CML_CH22 = 0
DAC_CMLSCALE_CH22 = 0
DMON_ENA_CH22 = 0
DMON_SW_CH22 = 0
TDCTEST_CH22 = 1
AMON_CTRL_CH22 = 0
COMP_SPI_CH22 = 0
DAC_SIPM_SC_CH22 = 0
DAC_SIPM_CH22 = f
DAC_TTHRESH_SC_CH22 = 0
DAC_TTHRESH_CH22 = 20
DAC_AMPCOM_SC_CH22 = 2
DAC_AMPCOM_CH22 = 0
DAC_INPUTBIAS_SC_CH22 = 0
DAC_INPUTBIAS_CH22 = 4
DAC_ETHRESH_CH22 = 0
DAC_EBIAS_CH22 = 0
DAC_POLE_SC_CH22 = 0
DAC_POLE_CH22 = 3f
DAC_CML_CH22 = 8
DAC_DELAY_CH22 = 0
DAC_DELAY_BIT1_CH22 = 0
DAC_CHANNEL_MASK_CH22 = 0
RECV_ALL_CH22 = 1
ANODE_FLAG_CH23 = 1
CATHODE_FLAG_CH23 = 1
S_SWITCH_CH23 = 1
SORD_CH23 = 0
SORD_NOT_CH23 = 1
EDGE_CH23 = 1
EDGE_CML_CH23 = 0
DAC_CMLSCALE_CH23 = 0
DMON_ENA_CH23 = 0
DMON_SW_CH23 = 0
TDCTEST_CH23 = 1
AMON_CTRL_CH23 = 0
COMP_SPI_CH23 = 0
DAC_SIPM_SC_CH23 = 0
DAC_SIPM_CH23 = f
DAC_TTHRESH_SC_CH23 = 0
DAC_TTHRESH_CH23 = 20
DAC_AMPCOM_SC_CH23 = 2
DAC_AMPCOM_CH23 = 0
DAC_INPUTBIAS_SC_CH23 = 0
DAC_INPUTBIAS_CH23 = 4
DAC_ETHRESH_CH23 = 0
DAC_EBIAS_CH23 = 0
DAC_POLE_SC_CH23 = 0
DAC_POLE_CH23 = 3f
DAC_CML_CH23 = 8
DAC_DELAY_CH23 = 0
DAC_DELAY_BIT1_CH23 = 0
DAC_CHANNEL_MASK_CH23 = 0
RECV_ALL_CH23 = 1
ANODE_FLAG_CH24 = 1
CATHODE_FLAG_CH24 = 1
S_SWITCH_CH24 = 1
SORD_CH24 = 0
SORD_NOT_CH24 = 1
EDGE_CH24 = 1
EDGE_CML_CH24 = 0
DAC_CMLSCALE_CH24 = 0
DMON_ENA_CH24 = 0
DMON_SW_CH24 = 0
TDCTEST_CH24 = 1
AMON_CTRL_CH24 = 0
COMP_SPI_CH24 = 0
DAC_SIPM_SC_CH24 = 0
DAC_SIPM_CH24 = f
DAC_TTHRESH_SC_CH24 = 0
DAC_TTHRESH_CH24 = 20
DAC_AMPCOM_SC_CH24 = 2
DAC_AMPCOM_CH24 = 0
DAC_INPUTBIAS_SC_CH24 = 0
DAC_INPUTBIAS_CH24 = 4
DAC_ETHRESH_CH24 = 0
DAC_EBIAS_CH24 = 0
DAC_POLE_SC_CH24 = 0
DAC_POLE_CH24 = 3f
DAC_CML_CH24 = 8
DAC_DELAY_CH24 = 0
DAC_DELAY_BIT1_CH24 = 0
DAC_CHANNEL_MASK_CH24 = 0
RECV_ALL_CH24 = 1
ANODE_FLAG_CH25 = 1
CATHODE_FLAG_CH25 = 1
S_SWITCH_CH25 = 1
SORD_CH25 = 0
SORD_NOT_CH25 = 1
EDGE_CH25 = 1
EDGE_CML_CH25 = 0
DAC_CMLSCALE_CH25 = 0
DMON_ENA_CH25 = 0
DMON_SW_CH25 = 0
TDCTEST_CH25 = 1
AMON_CTRL_CH25 = 0
COMP_SPI_CH25 = 0
DAC_SIPM_SC_CH25 = 0
DAC_SIPM_CH25 = f
DAC_TTHRESH_SC_CH25 = 0
DAC_TTHRESH_CH25 = 20
DAC_AMPCOM_SC_CH25 = 2
DAC_AMPCOM_CH25 = 0
DAC_INPUTBIAS_SC_CH25 = 0
DAC_INPUTBIAS_CH25 = 4
DAC_ETHRESH_CH25 = 0
DAC_EBIAS_CH25 = 0
DAC_POLE_SC_CH25 = 0
DAC_POLE_CH25 = 3f
DAC_CML_CH25 = 8
DAC_DELAY_CH25 = 0
DAC_DELAY_BIT1_CH25 = 0
DAC_CHANNEL_MASK_CH25 = 0
RECV_ALL_CH25 = 1
ANODE_FLAG_CH26 = 1
CATHODE_FLAG_CH26 = 1
S_SWITCH_CH26 = 1
SORD_CH26 = 0
SORD_NOT_CH26 = 1
EDGE_CH26 = 1
EDGE_CML_CH26 = 0
DAC_CMLSCALE_CH26 = 0
DMON_ENA_CH26 = 0
DMON_SW_CH26 = 0
TDCTEST_CH26 = 1
AMON_CTRL_CH26 = 0
COMP_SPI_CH26 = 0
DAC_SIPM_SC_CH26 = 0
DAC_SIPM_CH26 = f
DAC_TTHRESH_SC_CH26 = 0
DAC_TTHRESH_CH26 = 20
DAC_AMPCOM_SC_CH26 = 2
DAC_AMPCOM_CH26 = 0
DAC_INPUTBIAS_SC_CH26 = 0
DAC_INPUTBIAS_CH26 = 4
DAC_ETHRESH_CH26 = 0
DAC_EBIAS_CH26 = 0
DAC_POLE_SC_CH26 = 0
DAC_POLE_CH26 = 3f
DAC_CML_CH26 = 8
DAC_DELAY_CH26 = 0
DAC_DELAY_BIT1_CH26 = 0
DAC_CHANNEL_MASK_CH26 = 0
RECV_ALL_CH26 = 1
ANODE_FLAG_CH27 = 1
CATHODE_FLAG_CH27 = 1
S_SWITCH_CH27 = 1
SORD_CH27 = 0
SORD_NOT_CH27 = 1
EDGE_CH27 = 1
EDGE_CML_CH27 = 0
DAC_CMLSCALE_CH27 = 0
DMON_ENA_CH27 = 0
DMON_SW_CH27 = 0
TDCTEST_CH27 = 1
AMON_CTRL_CH27 = 0
COMP_SPI_CH27 = 0
DAC_SIPM_SC_CH27 = 0
DAC_SIPM_CH27 = f
DAC_TTHRESH_SC_CH27 = 0
DAC_TTHRESH_CH27 = 20
DAC_AMPCOM_SC_CH27 = 2
DAC_AMPCOM_CH27 = 0
DAC_INPUTBIAS_SC_CH27 = 0
DAC_INPUTBIAS_CH27 = 4
DAC_ETHRESH_CH27 = 0
DAC_EBIAS_CH27 = 0
DAC_POLE_SC_CH27 = 0
DAC_POLE_CH27 = 3f
DAC_CML_CH27 = 8
DAC_DELAY_CH27 = 0
DAC_DELAY_BIT1_CH27 = 0
DAC_CHANNEL_MASK_CH27 = 0
RECV_ALL_CH27 = 1
ANODE_FLAG_CH28 = 1
CATHODE_FLAG_CH28 = 1
S_SWITCH_CH28 = 1
SORD_CH28 = 0
SORD_NOT_CH28 = 1
EDGE_CH28 = 1
EDGE_CML_CH28 = 0
DAC_CMLSCALE_CH28 = 0
DMON_ENA_CH28 = 0
DMON_SW_CH28 = 0
TDCTEST_CH28 = 1
AMON_CTRL_CH28 = 0
COMP_SPI_CH28 = 0
DAC_SIPM_SC_CH28 = 0
DAC_SIPM_CH28 = f
DAC_TTHRESH_SC_CH28 = 0
DAC_TTHRESH_CH28 = 20
DAC_AMPCOM_SC_CH28 = 2
DAC_AMPCOM_CH28 = 0
DAC_INPUTBIAS_SC_CH28 = 0
DAC_INPUTBIAS_CH28 = 4
DAC_ETHRESH_CH28 = 0
DAC_EBIAS_CH28 = 0
DAC_POLE_SC_CH28 = 0
DAC_POLE_CH28 = 3f
DAC_CML_CH28 = 8
DAC_DELAY_CH28 = 0
DAC_DELAY_BIT1_CH28 = 0
DAC_CHANNEL_MASK_CH28 = 0
RECV_ALL_CH28 = 1
ANODE_FLAG_CH29 = 1
CATHODE_FLAG_CH29 = 1
S_SWITCH_CH29 = 1
SORD_CH29 = 0
SORD_NOT_CH29 = 1
EDGE_CH29 = 1
EDGE_CML_CH29 = 0
DAC_CMLSCALE_CH29 = 0
DMON_ENA_CH29 = 0
DMON_SW_CH29 = 0
TDCTEST_CH29 = 1
AMON_CTRL_CH29 = 0
COMP_SPI_CH29 = 0
DAC_SIPM_SC_CH29 = 0
DAC_SIPM_CH29 = f
DAC_TTHRESH_SC_CH29 = 0
DAC_TTHRESH_CH29 = 20
DAC_AMPCOM_SC_CH29 = 2
DAC_AMPCOM_CH29 = 0
DAC_INPUTBIAS_SC_CH29 = 0
DAC_INPUTBIAS_CH29 = 4
DAC_ETHRESH_CH29 = 0
DAC_EBIAS_CH29 = 0
DAC_POLE_SC_CH29 = 0
DAC_POLE_CH29 = 3f
DAC_CML_CH29 = 8
DAC_DELAY_CH29 = 0
DAC_DELAY_BIT1_CH29 = 0
DAC_CHANNEL_MASK_CH29 = 0
RECV_ALL_CH29 = 1
ANODE_FLAG_CH30 = 1
CATHODE_FLAG_CH30 = 1
S_SWITCH_CH30 = 1
SORD_CH30 = 0
SORD_NOT_CH30 = 1
EDGE_CH30 = 1
EDGE_CML_CH30 = 0
DAC_CMLSCALE_CH30 = 0
DMON_ENA_CH30 = 0
DMON_SW_CH30 = 0
TDCTEST_CH30 = 1
AMON_CTRL_CH30 = 0
COMP_SPI_CH30 = 0
DAC_SIPM_SC_CH30 = 0
DAC_SIPM_CH30 = f
DAC_TTHRESH_SC_CH30 = 0
DAC_TTHRESH_CH30 = 20
DAC_AMPCOM_SC_CH30 = 2
DAC_AMPCOM_CH30 = 0
DAC_INPUTBIAS_SC_CH30 = 0
DAC_INPUTBIAS_CH30 = 4
DAC_ETHRESH_CH30 = 0
DAC_EBIAS_CH30 = 0
DAC_POLE_SC_CH30 = 0
DAC_POLE_CH30 = 3f
DAC_CML_CH30 = 8
DAC_DELAY_CH30 = 0
DAC_DELAY_BIT1_CH30 = 0
DAC_CHANNEL_MASK_CH30 = 0
RECV_ALL_CH30 = 1
ANODE_FLAG_CH31 = 1
CATHODE_FLAG_CH31 = 1
S_SWITCH_CH31 = 1
SORD_CH31 = 0
SORD_NOT_CH31 = 1
EDGE_CH31 = 1
EDGE_CML_CH31 = 0
DAC_CMLSCALE_CH31 = 0
DMON_ENA_CH31 = 0
DMON_SW_CH31 = 0
TDCTEST_CH31 = 1
AMON_CTRL_CH31 = 0
COMP_SPI_CH31 = 0
DAC_SIPM_SC_CH31 = 0
DAC_SIPM_CH31 = f
DAC_TTHRESH_SC_CH31 = 0
DAC_TTHRESH_CH31 = 20
DAC_AMPCOM_SC_CH31 = 2
DAC_AMPCOM_CH31 = 0
DAC_INPUTBIAS_SC_CH31 = 0
DAC_INPUTBIAS_CH31 = 4
DAC_ETHRESH_CH31 = 0
DAC_EBIAS_CH31 = 0
DAC_POLE_SC_CH31 = 0
DAC_POLE_CH31 = 3f
DAC_CML_CH31 = 8
DAC_DELAY_CH31 = 0
DAC_DELAY_BIT1_CH31 = 0
DAC_CHANNEL_MASK_CH31 = 0
RECV_ALL_CH31 = 1
DAC_TDC_VND2C_SCALE = 0
DAC_TDC_VND2C_OFFSET = 3
DAC_TDC_VND2C = 1f
DAC_TDC_VNCntBuffer_SCALE = 0
DAC_TDC_VNCntBuffer_OFFSET = 3
DAC_TDC_VNCntBuffer = 7
DAC_TDC_VNCnt_SCALE = 0
DAC_TDC_VNCnt_OFFSET = 3
DAC_TDC_VNCnt = 28
DAC_TDC_VNPCP_SCALE = 0
DAC_TDC_VNPCP_OFFSET = 3
DAC_TDC_VNPCP = 18
DAC_TDC_VNVCODELAY_SCALE = 0
DAC_TDC_VNVCODELAY_OFFSET = 3
DAC_TDC_VNVCODELAY = a
DAC_TDC_VNVCOBUFFER_SCALE = 0
DAC_TDC_VNVCOBUFFER_OFFSET = 0
DAC_TDC_VNVCOBUFFER = 0
DAC_TDC_VNHITLOGIC_SCALE = 0
DAC_TDC_VNHITLOGIC_OFFSET = 3
DAC_TDC_VNHITLOGIC = 20
DAC_TDC_VNPFC_SCALE = 0
DAC_TDC_VNPFC_OFFSET = 3
DAC_TDC_VNPFC = 8
DAC_TDC_LATCHBIAS = 708
COIN_XBAR_lower_RX_ena = 0
COIN_XBAR_lower_TX_ena = 0
COIN_XBAR_lower_TX_vDAC = 0
COIN_XBAR_lower_TX_iDAC = 0
COIN_MAT_XBL = 0
COIN_MAT_CH0 = 0
COIN_MAT_CH1 = 0
COIN_MAT_CH2 = 0
COIN_MAT_CH3 = 0
COIN_MAT_CH4 = 0
COIN_MAT_CH5 = 0
COIN_MAT_CH6 = 0
COIN_MAT_CH7 = 0
COIN_MAT_CH8 = 0
COIN_MAT_CH9 = 0
COIN_MAT_CH10 = 0
COIN_MAT_CH11 = 0
COIN_MAT_CH12 = 0
COIN_MAT_CH13 = 0
COIN_MAT_CH14 = 0
COIN_MAT_CH15 = 0
COIN_MAT_CH16 = 0
COIN_MAT_CH17 = 0
COIN_MAT_CH18 = 0
COIN_MAT_CH19 = 0
COIN_MAT_CH20 = 0
COIN_MAT_CH21 = 0
COIN_MAT_CH22 = 0
COIN_MAT_CH23 = 0
COIN_MAT_CH24 = 0
COIN_MAT_CH25 = 0
COIN_MAT_CH26 = 0
COIN_MAT_CH27 = 0
COIN_MAT_CH28 = 0
COIN_MAT_CH29 = 0
COIN_MAT_CH30 = 0
COIN_MAT_CH31 = 0
COIN_MAT_XBH = 0
COIN_XBAR_upper_RX_ena = 0
COIN_XBAR_upper_TX_ena = 0
COIN_XBAR_upper_TX_vDAC = 0
COIN_XBAR_upper_TX_iDAC = 0
COIN_WND = 0
AMON_EN = 0
AMON_DAC = 0
DIG_MON1_EN = 0
DIG_MON1_DAC = dc
DIG_MON2_EN = 0
DIG_MON2_DAC = dc
LVDS_TX_VCM = 9b
LVDS_TX_BIAS = 0
*/
unsigned char config_DCR[] = {
0x3,0x0,0x0,0xb8,0x21,0xc0,0x3,0x8,0x80,
0x0,0xc0,0x1f,0x78,0x43,0x80,0x7,0x10,0x0,0x1,
0x80,0x3f,0xf0,0x86,0x0,0xf,0x20,0x0,0x2,0x0,
0x7f,0xe0,0xd,0x1,0x1e,0x40,0x0,0x4,0x0,0xfe,
0xc0,0x1b,0x2,0x3c,0x80,0x0,0x8,0x0,0xfc,0x81,
0x37,0x4,0x78,0x0,0x1,0x10,0x0,0xf8,0x3,0x6f,
0x8,0xf0,0x0,0x2,0x20,0x0,0xf0,0x7,0xde,0x10,
0xe0,0x1,0x4,0x40,0x0,0xe0,0xf,0xbc,0x21,0xc0,
0x3,0x8,0x80,0x0,0xc0,0x1f,0x78,0x43,0x80,0x7,
0x10,0x0,0x1,0x80,0x3f,0xf0,0x86,0x0,0xf,0x20,
0x0,0x2,0x0,0x7f,0xe0,0xd,0x1,0x1e,0x40,0x0,
0x4,0x0,0xfe,0xc0,0x1b,0x2,0x3c,0x80,0x0,0x8,
0x0,0xfc,0x81,0x37,0x4,0x78,0x0,0x1,0x10,0x0,
0xf8,0x3,0x6f,0x8,0xf0,0x0,0x2,0x20,0x0,0xf0,
0x7,0xde,0x10,0xe0,0x1,0x4,0x40,0x0,0xe0,0xf,
0xbc,0x21,0xc0,0x3,0x8,0x80,0x0,0xc0,0x1f,0x78,
0x43,0x80,0x7,0x10,0x0,0x1,0x80,0x3f,0xf0,0x86,
0x0,0xf,0x20,0x0,0x2,0x0,0x7f,0xe0,0xd,0x1,
0x1e,0x40,0x0,0x4,0x0,0xfe,0xc0,0x1b,0x2,0x3c,
0x80,0x0,0x8,0x0,0xfc,0x81,0x37,0x4,0x78,0x0,
0x1,0x10,0x0,0xf8,0x3,0x6f,0x8,0xf0,0x0,0x2,
0x20,0x0,0xf0,0x7,0xde,0x10,0xe0,0x1,0x4,0x40,
0x0,0xe0,0xf,0xbc,0x21,0xc0,0x3,0x8,0x80,0x0,
0xc0,0x1f,0x78,0x43,0x80,0x7,0x10,0x0,0x1,0x80,
0x3f,0xf0,0x86,0x0,0xf,0x20,0x0,0x2,0x0,0x7f,
0xe0,0xd,0x1,0x1e,0x40,0x0,0x4,0x0,0xfe,0xc0,
0x1b,0x2,0x3c,0x80,0x0,0x8,0x0,0xfc,0x81,0x37,
0x4,0x78,0x0,0x1,0x10,0x0,0xf8,0x3,0x6f,0x8,
0xf0,0x0,0x2,0x20,0x0,0xf0,0x7,0xde,0x10,0xe0,
0x1,0x4,0x40,0x0,0xe0,0xf,0xb4,0x6f,0xdc,0x85,
0xd,0x53,0x0,0x1c,0x98,0x40,0x38,0x0,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,
0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x3b,0x76,0xb2,
0x1};

