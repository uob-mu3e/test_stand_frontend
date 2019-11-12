-- Register Map
-- Note: 
-- For mupix_registers.tcl script to identify write register, use naming scheme: 		***_REGISTER_W
-- For mupix_registers.tcl script to identify read  register, use naming scheme: 		***_REGISTER_R
-- For mupix_registers.tcl script to identify bit range     , use naming scheme: 		***_RANGE
-- For mupix_registers.tcl script to identify single bit constant, use naming scheme: 	***_BIT

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;

package mupix_registers is

--////////////////////////////////////////////--
--/////////////WRITE REGISTER MAP/////////////--
--////////////////////////////////////////////--
	constant LED_REGISTER_W										:  integer := 16#00#;
		constant USE_EXTERNAL_RESET_BIT						: 	integer := 16;
		constant A_SPARE_CLOCK_RESET_BIT						: 	integer := 20;
		constant B_SPARE_CLOCK_RESET_BIT						: 	integer := 24;
		
	
	constant RESET_REGISTER_W									:  integer := 16#01#;
		constant RESET_ALL_BIT									:  integer := 0;
		constant RESET_RO_BIT									:  integer := 1;
		constant RESET_PLL_BIT									:  integer := 2;
		constant RESET_DDR3_GLOBAL_BIT						:  integer := 3;	-- formerly 11
		constant RESET_DDR3_SOFT_BIT							:  integer := 4;	-- formerly 15			
		constant RESET_SPI_BIT									:	integer := 5;	-- formerly 4
		constant RESET_INJECTIONS_BIT							:	integer := 6;	-- formerly 5
		constant RESET_ROMEMWRITER_BIT						:  integer := 7;	-- formerly 6
		constant RESET_COUNTER_BIT								:  integer := 8;	 
		constant RESET_PCIE_BIT									:  integer := 9;
		constant RESET_TRIGGER_BIT								:  integer := 10;
		constant RESET_HISTOGRAMS_BIT							:	integer := 11;	-- formerly 20	
		constant RESET_DATA_GEN_BIT							: 	integer := 12;	-- formerly 23		
		constant RESET_DATA_GEN_SYNCRES_BIT					:	integer := 13;	-- formerly 27	
		constant RESET_MP8_SLOWCTRL_BIT						:  integer := 14;
		constant RESET_RECEIVER_BIT							: 	integer := 16;	-- single reset for all channels!
		constant RESET_LVDS_BIT									:	integer := 17;	-- single reset for all channels!
		-- syncreset per chip	
		constant RESET_A_FRONT_BIT								:  integer := 20;
		constant RESET_A_BACK_BIT								:  integer := 21;
		constant RESET_B_FRONT_BIT								:  integer := 22;
		constant RESET_B_BACK_BIT								:  integer := 23;
		constant RESET_BERT_DATAGEN_BIT						:	integer := 24;
		constant RESET_BERT_ALIGN_BIT							:	integer := 25;
		constant RESET_ERRCOUNTER_BIT							: 	integer := 29;
		constant RESET_CPU_BIT									: 	integer := 30;
		constant RESET_MUPIX_BIT								:  integer := 31; -- single syncreset for all!
																								-- set to '1' for a global FPGA master slave reset!	

	constant CFG_PLL_REGISTER_W			: integer := 16#02#;			
	-- reconfiguration of frequency f = (M/N) * 125MHz, datarate = f * 10 Mbit/s
		subtype PLL_DIVISOR_W_RANGE	is integer range 7 downto 0;	
		-- set to M
		subtype PLL_DIVIDEND_W_RANGE	is integer range 15 downto 8;
		-- set to N

	constant DDR3_REGISTER_W				:	integer := 16#03#;			
	-- for polling devices, use DDR3 buffer on devboard: 0x6001
		constant DDR3_ENABLE_BIT			:	integer := 0;					
		-- set to '1'
		subtype	DDR3_RD_SLOWDOWN_RANGE 	is integer range 31 downto 4; 
		-- set to 0x600
		
	constant RECEIVER_REGISTER_W			:  integer := 16#04#; 
		constant GX_LVDS_MUX_BIT			: integer := 0;					
		-- '0' for GX, '1' for LVDS
		constant LVDS_ORDER_BIT 			: integer := 1;					
		-- '0': MSB...LSB, '1': LSB...MSB
		subtype GX_REGISTER_MUX_RANGE 	is integer range 3 downto 2;	
		constant GX_LOOPBACK_BIT			: integer := 4;
		
		subtype GX_BERT_ENA_RANGE 		is integer range 15 downto 8;	
		subtype GX_BERT_ERRINJ_RANGE 	is integer range 23 downto 16;	
		-- selects runcounter + errcounter channels:
		-- "00": ch 0,4,8,12
		-- "01": ch 1,5,9,13
		-- "10": ch 2,6,10,14
		-- "11": ch 3,7,11,15
		-- also selects FREQLOCKED + SYNCSTATUS
		-- "00": FREQLOCKED + SYNCSTATUS
		-- "01": PLL_LOCKED + ERR_OUT
		-- "10": PATTERNDETECT + DISPERR
		-- "11": FREQLOCKED + SYNCSTATUS
		
	constant RO_MODE_REGISTER_W			:  integer := 16#05#;
	-- select the RO mode
		subtype RO_MODE_RANGE 				is integer range 2 downto 0;
		-- set to 0 for raw data from links 3 downto 0
		-- set to 1 for single link readout, selected with RO_MODE_SINGLECHIP_RANE
		-- set to 2 for single link readout zero suppressed
		-- set to 3 for telescope RO using data sorting and packet building
		-- set to 4 for multilink RO using simple round robin arbitration (rate limited!)
		-- RO modes 3 and 4 include hitbus + trigger informations
		constant RO_ENABLE_BIT 				: 	integer := 4;
		-- set to '1' to enable the writing of readout data to memory
		-- set to '0' to disable memory writing
		--subtype RO_MODE_SINGLECHIP_RANGE is integer range NLVDS_LOG2-1+8 downto 8;
		-- chose link (0 to 31)
		subtype RO_IS_ATLASPIX_RANGE is integer range 31 downto 16;
		
	constant RO_CHIPMARKER_REGISTER_W		: integer := 16#06#;
	-- defines the marker indicating hits
		subtype RO_HITMARKER_RANGE				is integer range 7 downto 2;
		-- global marker for telescope RO
		subtype RO_CHIPMARKER_A_FRONT_RANGE is integer range 7 downto 0;
		-- marker for single chip RO
		subtype RO_CHIPMARKER_A_BACK_RANGE 	is integer range 15 downto 8;	
		subtype RO_CHIPMARKER_B_FRONT_RANGE is integer range 23 downto 16;	
		subtype RO_CHIPMARKER_B_BACK_RANGE	is integer range 31 downto 24;
		-- markers for multilink RO have to be defined!
		
	constant RO_PRESCALER_REGISTER_W							: integer := 16#07#;
	-- allows empty frames to pass the zero suppression
		subtype RO_PRESCALER_RANGE is integer range 31 downto 0;		
		-- this number N empty frames will be dropped before an empty frame can pass the zero suppression
		-- rate of empty frames is reduced by this number
		-- does not affect frames with hits
		
	constant DATA_GEN_REGISTER_W		: integer := 16#08#;		-- former DATA FAKER
	-- MuPix8 data generator
		constant MUX_DATA_GEN_BIT				: integer := 0;
		constant DATA_GEN_USE_CLKREF_BIT		: integer := 1;
		constant DATA_GEN_SENDCOUNTER_BIT	: integer := 2;
		
		subtype DATA_GEN_CKDIVEND_RANGE		is integer range 8 downto 3;		-- 6 bit
		subtype DATA_GEN_CKDIVEND2_RANGE	   is integer range 14 downto 9;		-- 6 bit
		subtype DATA_GEN_TSPHASE_RANGE		is integer range 20 downto 15;	-- 6 bit
		subtype DATA_GEN_TIMEREND_RANGE		is integer range 24 downto 21;	-- 4 bit
		subtype DATA_GEN_SLOWDOWNEND_RANGE	is integer range 28 downto 25;	-- 4 bit
		subtype DATA_GEN_MODE_RANGE			is integer range 30 downto 29;   -- 2 bit
		constant PORT_READREG_BIT				: integer := 31;
		
	constant DATA_GEN_2_REGISTER_W			: integer :=16#09#;
		subtype DATA_GEN_MAXCYCEND_RANGE	is integer range 5 downto 0;			-- 6 bit
		subtype DATA_GEN_RESETCKDIVEND_RANGE is integer range 9 downto 6;		-- 4 bit
		subtype DATA_GEN_LINKSEL_RANGE		is integer range 11 downto 10;	-- 2 bit
		subtype SET_HIT_NUM_RANGE 			is integer range 17 downto 12; 		-- same hit num for all three submatrices
		constant PORTA_WRITEREG_BIT			: integer := 30;						
		constant PORTB_WRITEREG_BIT			: integer := 31;						
		-- bits 29 downto 18 unused
		
--	constant HITSORTER_CNTDOWN_REGISTER_W	: integer := 16#0A#;
--	-- Hitsorter register: defines the number of # of empty frames which lead to sending of a packet
--	-- @ AK: please correct me if I'm wrong here
--		subtype HITSORTER_CNTDOWN_RANGE is integer range 7 downto 0;
		
--	constant HITSORTER_INIT_REGISTER_W		: integer := 16#0B#;--21;
--	-- Initialize the hitsorter memory for debugging
--		constant READINITHITSORTMEM_BIT 		: integer := 0;
--		constant WRITEINITHITSORTMEM_BIT 	: integer := 1;	

--	constant HITSORTER_EVENTTIME_SEL_REGISTER_W		: integer := 16#0C#;
--	-- Set to '0' if MuPix counter is to be used for sorter
--	-- set to '1' if FPGA 125 MHz counter is to be used for sorter
--	-- @ AK: please correct me if I'm wrong
--		constant HITSORTER_COARSETIME_SEL_BIT	: integer := 0;	

	constant HITSORTER_TIMING_REGISTER_W	: integer := 16#0A#;		-- same register as HITSORTER_CNTDOWN_REGISTER_W
		subtype HITSORTER_TSDELAY_RANGE		is integer range  9 downto  0;	-- 10b
		subtype HITSORTER_TSDIVIDER_RANGE	is integer range 15 downto 12;
		
	constant CONTROL_SIGNALS_REGISTER_W : integer := 16#0B#;
		constant TRIGGER_OUT_A_FRONT : integer := 0;
-- below: normal telescope
		constant TRIGGER_OUT_A_BACK : integer 	:= 1;
		constant TRIGGER_OUT_B_FRONT : integer := 2;
		constant TRIGGER_OUT_B_BACK : integer 	:= 3;
-- below: FEB
--		constant TRIGGER_OUT_B_FRONT : integer := 1;
--		constant TRIGGER_OUT_C_FRONT : integer := 2;
--		constant TRIGGER_OUT_E_FRONT : integer := 3;
		
		constant LDCOL_OUT_A_FRONT : integer 	:= 4;
		constant LDCOL_OUT_A_BACK : integer 	:= 5;
		constant LDCOL_OUT_B_FRONT : integer 	:= 6;
		constant LDCOL_OUT_B_BACK : integer 	:= 7;
		constant RDCOL_OUT_A_FRONT : integer 	:= 8;
		constant RDCOL_OUT_A_BACK : integer 	:= 9;
		constant RDCOL_OUT_B_FRONT : integer 	:= 10;
		constant RDCOL_OUT_B_BACK : integer 	:= 11;
		constant PD_OUT_A_FRONT : integer 		:= 12;
		constant PD_OUT_A_BACK : integer 		:= 13;
		constant PD_OUT_B_FRONT : integer 		:= 14;
		constant PD_OUT_B_BACK : integer 		:= 15;
		constant LDPIX_OUT_A_FRONT : integer 	:= 16;
		constant LDPIX_OUT_A_BACK : integer 	:= 17;
		constant LDPIX_OUT_B_FRONT : integer 	:= 18;
		constant LDPIX_OUT_B_BACK : integer 	:= 19;
	
	constant ADAPTERCARD_DEBUG_REGISTER_W			: integer := 16#0D#;
		-- for RJ45 loopback --> test in/out
		-- write bit 0 --> read 16#0D# bit 0 --> expect RST_LOCAL_BIT to change
		-- write bit 1 --> read 16#0D# bit 1 --> expect TRG_LOCAL_BIT to change
		constant CLK_LOCAL_BIT 		: integer := 0;
		constant BUSY_LOCAL_BIT 	: integer := 1;
		
	constant SANTALUZ1_REGISTER_W : integer := 16#0E#;
		subtype SL_DISABLE_RANGE	is integer range 7 downto 0;
		subtype SL_RATESEL_RANGE	is integer range 15 downto 8;
		subtype SL_TXLED_RANGE	is integer range 23 downto 16;
		subtype SL_RXLED_RANGE	is integer range 31 downto 24;
		
	constant SANTALUZ2_REGISTER_W : integer := 16#0F#;
		subtype SL_STATLED_RANGE	is integer range 7 downto 0;
		
--	constant TOT_GENERATOR_REGISTER_W				: integer := 16#0E#;
--		subtype CTR_HI_RANGE is integer range 27 downto 0;					-- high for x clockcycles
--		
--	constant TOT_GENERATOR_2_REGISTER_W				: integer := 16#0F#;
--		subtype CTR_LO_RANGE is integer range 27 downto 0; 				-- low for x clockcycles

	constant DAC_WRITE_REGISTER_W			: integer := 16#10#;
	-- BOARD DACS and ADC write request
	-- not regwritten signal anymore as we have 4 DACs daisy-chained
		subtype DAC_WRITE_A_FRONT_RANGE 	is integer range 2 downto 0;
		-- bit 0: write 4xDACs: Injection, ThPix, ThLow, ThHigh
		-- bit 1: write TempDAC
		-- bit 2: write TempADC
-- below: normal telescope
		subtype DAC_WRITE_A_BACK_RANGE 	is integer range 6 downto 4;
		subtype DAC_WRITE_B_FRONT_RANGE 	is integer range 10 downto 8;
		subtype DAC_WRITE_B_BACK_RANGE 	is integer range 14 downto 12;
-- below: FEB telescope
--		subtype DAC_WRITE_B_FRONT_RANGE 	is integer range 6 downto 4;
--		subtype DAC_WRITE_C_FRONT_RANGE	is integer range 10 downto 8;
--		subtype DAC_WRITE_E_FRONT_RANGE 	is integer range 14 downto 12;
		subtype DAC_READ_SELECT_RANGE 	is integer range 17 downto 16;
	
	constant THRESHOLD_DAC_A_FRONT_REGISTER_W			:	integer := 16#11#;
	-- contains ThLow and ThHigh
	constant THRESHOLD_DAC_A_BACK_REGISTER_W			:	integer := 16#12#;
	constant THRESHOLD_DAC_B_FRONT_REGISTER_W			:	integer := 16#13#;
	constant THRESHOLD_DAC_B_BACK_REGISTER_W			:	integer := 16#14#;
	-- we just use the same registers for A and C and B and E,
	-- just make sure to use the correct values in software!
	constant THRESHOLD_DAC_C_FRONT_REGISTER_W			:	integer := 16#11#;
	constant THRESHOLD_DAC_C_BACK_REGISTER_W			:	integer := 16#12#;
	constant THRESHOLD_DAC_E_FRONT_REGISTER_W			:	integer := 16#13#;
	constant THRESHOLD_DAC_E_BACK_REGISTER_W			:	integer := 16#14#;
		subtype THRESHOLD_LOW_RANGE 	is integer range 15 downto 0;
		subtype THRESHOLD_HIGH_RANGE 	is integer range 31 downto 16;		
	
	constant INJECTION_DAC_A_FRONT_REGISTER_W			:	integer := 16#15#;
	-- containts Injection and ThPix DAC
	constant INJECTION_DAC_A_BACK_REGISTER_W			:	integer := 16#16#;
	constant INJECTION_DAC_B_FRONT_REGISTER_W			:	integer := 16#17#;
	constant INJECTION_DAC_B_BACK_REGISTER_W			:	integer := 16#18#;
	-- we just use the same registers for A and C and B and E,
	-- just make sure to use the correct values in software!
	constant INJECTION_DAC_C_FRONT_REGISTER_W			:	integer := 16#15#;
	constant INJECTION_DAC_C_BACK_REGISTER_W			:	integer := 16#16#;
	constant INJECTION_DAC_E_FRONT_REGISTER_W			:	integer := 16#17#;
	constant INJECTION_DAC_E_BACK_REGISTER_W			:	integer := 16#18#;
		subtype INJECTION1_RANGE 		is integer range 15 downto 0;
		subtype THRESHOLD_PIX_RANGE 	is integer range 31 downto 16;
	
	constant TEMP_A_FRONT_REGISTER_W					:	integer := 16#19#;
	-- containts TEMP DAC and TEMP ADC
	constant TEMP_A_BACK_REGISTER_W					:	integer := 16#1A#;
	constant TEMP_B_FRONT_REGISTER_W					:	integer := 16#1B#;
	constant TEMP_B_BACK_REGISTER_W					:	integer := 16#1C#;
	-- we just use the same registers for A and C and B and E,
	-- just make sure to use the correct values in software!
	constant TEMP_C_FRONT_REGISTER_W					:	integer := 16#19#;
	constant TEMP_C_BACK_REGISTER_W					:	integer := 16#1A#;
	constant TEMP_E_FRONT_REGISTER_W					:	integer := 16#1B#;
	constant TEMP_E_BACK_REGISTER_W					:	integer := 16#1C#;
		subtype TEMP_DAC_RANGE 		is integer range 15 downto 0;
		subtype TEMP_ADC_W_RANGE 	is integer range 31 downto 16;
--!!!--
--		subtype TEMP_ADC_RANGE_R_CFG is integer range 31 downto 0;		
--		subtype TEMP_ADC_RANGE_R_DATA is integer range 31 downto 0;							
		
	constant	INJECTION_DURATION_REGISTER_W 				: integer := 16#1D#;
	-- define Injection Duration and rate
		subtype INJECTION_DURATTION_RANGE 	is integer range 15 downto 0;
		subtype INJECTION_RATE_RANGE 			is integer range 31 downto 16;
		
	constant INJECTION_REGISTER_W 							: integer := 16#1E#;	
	-- set target # of injections, or enable infinit injection mode, or reset injection counter
	-- bit 0 to 3 are enables for the different ports 
		subtype INJECTION_REGISTER_TARGET_COUNT_RANGE 	is integer range 27 downto 8;
		constant INJECTION_INFINIT_BIT						: integer := 30;
		constant INJECTION_REGISTER_COUNTER_RESET_BIT	: integer := 31;
	
	constant SLOW_CONTROL_REGISTER_W							: integer := 16#1F#;
		-- blocks of 4x8 bits per chip 
		-- bit 0: Data In
		-- bit 1: Clock1
		-- bit 2: Clock2
		-- bit 3: Load
		-- bit 4: ReadBack
		-- bit 7 downto 5: not used
		constant SLOW_CONTROL_DIN_A_FRONT_BIT	: integer := 0;
		constant SLOW_CONTROL_CK1_A_FRONT_BIT	: integer := 1;
		constant SLOW_CONTROL_CK2_A_FRONT_BIT	: integer := 2;
		constant SLOW_CONTROL_LD_A_FRONT_BIT	: integer := 3;
		constant SLOW_CONTROL_RB_A_FRONT_BIT	: integer := 4;
-- for normal telescope
		constant SLOW_CONTROL_DIN_A_BACK_BIT	: integer := 8;
		constant SLOW_CONTROL_CK1_A_BACK_BIT	: integer := 9;
		constant SLOW_CONTROL_CK2_A_BACK_BIT	: integer := 10;
		constant SLOW_CONTROL_LD_A_BACK_BIT		: integer := 11;
		constant SLOW_CONTROL_RB_A_BACK_BIT		: integer := 12;		
		constant SLOW_CONTROL_DIN_B_FRONT_BIT	: integer := 16;
		constant SLOW_CONTROL_CK1_B_FRONT_BIT	: integer := 17;
		constant SLOW_CONTROL_CK2_B_FRONT_BIT	: integer := 18;
		constant SLOW_CONTROL_LD_B_FRONT_BIT	: integer := 19;
		constant SLOW_CONTROL_RB_B_FRONT_BIT	: integer := 20;
		constant SLOW_CONTROL_DIN_B_BACK_BIT	: integer := 24;
		constant SLOW_CONTROL_CK1_B_BACK_BIT	: integer := 25;
		constant SLOW_CONTROL_CK2_B_BACK_BIT	: integer := 26;
		constant SLOW_CONTROL_LD_B_BACK_BIT		: integer := 27;
		constant SLOW_CONTROL_RB_B_BACK_BIT		: integer := 28;
-- for FEB!
--		constant SLOW_CONTROL_DIN_B_FRONT_BIT	: integer := 8;
--		constant SLOW_CONTROL_CK1_B_FRONT_BIT	: integer := 9;
--		constant SLOW_CONTROL_CK2_B_FRONT_BIT	: integer := 10;
--		constant SLOW_CONTROL_LD_B_FRONT_BIT	: integer := 11;
--		constant SLOW_CONTROL_RB_B_FRONT_BIT	: integer := 12;
--		constant SLOW_CONTROL_DIN_C_FRONT_BIT	: integer := 16;
--		constant SLOW_CONTROL_CK1_C_FRONT_BIT	: integer := 17;
--		constant SLOW_CONTROL_CK2_C_FRONT_BIT	: integer := 18;
--		constant SLOW_CONTROL_LD_C_FRONT_BIT	: integer := 19;
--		constant SLOW_CONTROL_RB_C_FRONT_BIT	: integer := 20;
--		constant SLOW_CONTROL_DIN_E_FRONT_BIT	: integer := 24;
--		constant SLOW_CONTROL_CK1_E_FRONT_BIT	: integer := 25;
--		constant SLOW_CONTROL_CK2_E_FRONT_BIT	: integer := 26;
--		constant SLOW_CONTROL_LD_E_FRONT_BIT	: integer := 27;
--		constant SLOW_CONTROL_RB_E_FRONT_BIT	: integer := 28;


		constant SLOW_CONTROL_USE_FPGA_SC_BIT	: integer := 30;
		constant SLOW_CONTROL_MP8_START_BIT		: integer := 31;

	constant DEBUG_CHIP_SELECT_REGISTER_W		: integer := 16#20#;
	-- general purpose register
		subtype CHIP_SELECT_RANGE	is integer range 2 downto 0;
		-- selects pre-/postsort channel
		subtype HITSORTER_DEBUG_SELECT_RANGE is integer range 5 downto 4;
		-- hitsorter debugging -> @ AK details?
		subtype SLOW_CONTROL_CKDIVEND_RANGE is integer range 23 downto 8;
		subtype SELECT_DATAOUT_RANGE is integer range 25 downto 24;
		constant LOOPBACK_WMEM_BIT : integer := 31;
		-- set to '1' to loopback writable memory into the readable memory

		
--	constant HIT_COUNTER_REGISTER_W				: integer := 16#17#;
--		subtype CHIP_SELECT	is integer range 2 downto 0;
--		constant SHOW_INJECTION_COUNTERS 		: integer := 4;
--		constant SHOW_IDENTICAL_HITS				: integer := 8;
	
	constant HISTOGRAM_REGISTER_W								: integer := 16#21#;
	-- used for histograms, not yet implemented again
	-- function for set time of histo data taking
		constant HISTOGRAM_TAKEDATA_BIT				: integer := 0;
		constant HISTOGRAM_ZEROMEM_BIT				: integer := 1;
		subtype	HISTOGRAM_HITMAP_CHIPSEL_RANGE is integer range 3 downto 2;
		subtype	HISTOGRAM_SELECT_RANGE is integer range 11 downto 4;
		subtype 	HISTOGRAM_CKDIVEND_RANGE is integer range 15 downto 12;	
		subtype 	HISTOGRAM_READADDR_RANGE is integer range 31 downto 16;			
--		subtype  HISTOGRAM_KOMMA_READADDR_RANGE is integer range 19 downto 16;	
--		subtype  HISTOGRAM_COUNTER_READADDR_RANGE is integer range 20 downto 16;	
--		subtype  HISTOGRAM_ROW_READADDR_RANGE is integer range 23 downto 16;	
--		subtype  HISTOGRAM_TS_READADDR_RANGE is integer range 23 downto 16;			
--		subtype  HISTOGRAM_COL_READADDR_RANGE is integer range 31 downto 24;		
--		subtype	HISTOGRAM_MULTI_READADDR_RANGE is integer range 21 downto 16;
--		subtype	HISTOGRAM_TIMEDIFF_READADDR_RANGE is integer range 23 downto 16;
		
	constant LINK_REGISTER_W									: integer := 16#22#;
-- normal telescope	
		subtype LINK_SHARED_RANGE	is integer range 15 downto 0;
		subtype LINK_MASK_RANGE		is integer range 31 downto 16;		
		
	constant LINK_MASK_REGISTER_W									: integer := 16#23#;
-- below: FEB	
--		subtype LINK_MASK_RANGE		is integer range 31 downto 0;		
--	constant UNPACKER_CONFIG_REGISTER_W						: integer := 16#23#;
--		subtype TIMEREND_RANGE		is integer range 3 downto 0;
		
	-- -- for PSEUDO DATA GENERATOR:
	constant DATA_GEN_CHIP0_REGISTER_W							: integer := 16#24#;
	constant DATA_GEN_CHIP1_REGISTER_W							: integer := 16#25#;
	constant DATA_GEN_CHIP2_REGISTER_W							: integer := 16#26#;
	constant DATA_GEN_CHIP3_REGISTER_W							: integer := 16#27#;
		subtype SET_SLOWDOWN_RANGE is integer range 27 downto 0;  -- freq with which snapshots are taken
		--subtype SET_HIT_NUM_A_RANGE is integer range 23 downto 20; -- only 0-15 hits/submatrix
		--subtype SET_HIT_NUM_B_RANGE is integer range 27 downto 24;
		--subtype SET_HIT_NUM_C_RANGE is integer range 31 downto 28; -- all the same number of hits for now
		subtype DATA_GEN_LINKEN0_RANGE		is integer range 31 downto 28;	-- enable links of chip0
		subtype DATA_GEN_LINKEN1_RANGE		is integer range 31 downto 28;	-- enable links of chip1
		subtype DATA_GEN_LINKEN2_RANGE		is integer range 31 downto 28;	-- enable links of chip2
		subtype DATA_GEN_LINKEN3_RANGE		is integer range 31 downto 28;	-- enable links of chip3
		
	constant TIMESTAMP_GRAY_INVERT_REGISTER_W				: integer := 16#28#;
		constant TS_INVERT_BIT	: integer := 0;
		constant TS2_INVERT_BIT	: integer := 1;
		constant TS_GRAY_BIT		: integer := 2;
		constant TS2_GRAY_BIT	: integer := 3;
		
	constant INJECTION_PHASE_ADJUST_REGISTER_W			: integer := 16#29#;
		subtype SET_PHASE_RANGE	is integer range 7 downto 0;
		subtype MAX_PHASE_RANGE is integer range 15 downto 8;
		
	constant TRIGGER_REGISTER_W								: integer := 16#30#;
	-- control the TTL trigger inputs
		subtype TRIGGER_MASK_RANGE is integer range NTRIGGERS-1 downto 0;
		-- defines which signals actually produce a trigger
		subtype TRIGGER_VETO_RANGE is integer range 2*NTRIGGERS-1 downto NTRIGGERS;
		-- defines veto triggers
-- below: normal telescope		
		subtype TRIGGER_SCALER_SELECT_RANGE is integer range 2*NTRIGGERS+2 downto 2*NTRIGGERS;
-- below: FEB 		
--		subtype TRIGGER_SCALER_SELECT_RANGE is integer range 2*NTRIGGERS+1 downto 2*NTRIGGERS;
		-- 2 bits to select which trigger scaler to be displayed
		constant TRIGGER_INPUT_INVERT_BIT 				: integer := 29;		
		-- set to '1' to invert the trigger input, used for NIM/active low signals
		constant TRIGGER_REGISTER_SCALER_RESET_BIT		: integer := 30;
		constant TRIGGER_REGISTER_GENERATE_TRIGGER_BIT	: integer := 31;
		
	constant TRIGGER_WAIT_REGISTER_W							: integer := 16#31#;	
	-- only used with internal trigger generation
			
	constant TLU_ENABLE_REGISTER_W							: integer := 16#32#;		
		constant	TLU_ENABLE_BIT									: integer := 0;	
		-- set to '1' to enable TLU
	constant TLU_WAIT_DELAY_REGISTER_W						: integer := 16#33#;
	-- defines DELAY between trigger number received and busy signal disable
	-- for DESY Testbeams set to 0x <- find number from last testbeams!
	
	constant HITBUS_REGISTER_W						:	integer :=16#34#;
	-- control the hitbus
		constant HBENABLE_BIT						:	integer := 0;	
		-- set to '1' to enable HB
--		constant HISTO_TAKEDATA_BIT				: 	integer := 1;
--		constant HISTO_ZEROMEM_BIT					: 	integer := 2;
--		subtype HISTO_ADDRESS_RANGE	is integer range 13 downto 4;		
		constant HBINVERT_BIT						:	integer := 16;
		-- set to '1' if HB is active low
		
	constant NIOS_ADDRESS_REGISTER_W				: integer := 16#35#;
		subtype NIOS_ADDRESS_RANGE	is integer range 9 downto 0;
		constant NIOS_WR_BIT							: integer := 12;
		constant NIOS_RD_BIT							: integer := 16;
	constant NIOS_WRDATA_REGISTER_W				: integer := 16#36#;

	-- Registers above 0x38 are in use for the PCIe controller/DMA
	constant DMA_REGISTER_W										: integer := 16#38#;
	   constant DMA_ENABLE_BIT                         : integer := 0;
      constant DMA_NOW_BIT                            : integer := 1;

	constant DMA_CTRL_ADDR_LOW_REGISTER_W					: integer := 16#39#;
	constant DMA_CTRL_ADDR_HI_REGISTER_W					: integer := 16#3A#;
	constant DMA_DATA_ADDR_LOW_REGISTER_W					: integer := 16#3B#;
	constant DMA_DATA_ADDR_HI_REGISTER_W					: integer := 16#3C#;
	constant DMA_RAM_LOCATION_NUM_PAGES_W					: integer := 16#3D#;
		subtype DMA_RAM_LOCATION_RANGE 			is integer range 31 downto 20;
		subtype DMA_NUM_PAGES_RANGE				is integer range 19 downto 0;
	constant DMA_NUM_ADDRESSES_W								: integer := 16#3E#;
		subtype DMA_NUM_ADDRESSES_RANGE 	is integer range 11 downto 0; -- bits above 11 not used so far
	
	
------------------------------------------------------------------
---------------------- Read register Map -------------------------
------------------------------------------------------------------
	constant	PLL_REGISTER_R										: integer := 16#00#;
	-- state of PLL + DIPSWITCH
		subtype DIPSWITCH_RANGE is integer range 7 downto 0;
		-- dip 0 selects internal/external clock
		subtype PLL_LOCKED_RANGE is integer range 10 downto 8;
		constant PLL_LOCKED_BIT 								: integer := 8;	
		constant PLL_ACTIVE_BIT 								: integer := 11;
		-- '0' indicates CLOCK0 (internal) is selected
		-- '1' indicates CLOCK1 (external) is selected
		constant PLL_BAD0_BIT 									: integer := 12;	
		-- if '1': internal clock signal does not toggle -> that's bad!
		constant PLL_BAD1_BIT 									: integer := 13;		
		-- if '1': external clock signal does not toggle -> not connected?
		constant CFG_PLL_LOCKED_BIT							: integer := 16;
		-- '1': Reconfigurable PLL is locked
		constant CFG_PLL_REC_BUSY_BIT							: integer := 17;		
		-- '1': Reconfiguration is busy
		constant PCIE_PLL_LOCKED_BIT							: integer := 18;
		constant CLOCKGEN_PLL_LOCKED_BIT						: integer := 19;
		
	constant	VERSION_REGISTER_R								: integer := 16#01#;
		subtype VERSION_RANGE is integer range 31 downto 0;
		-- git hash
		
	constant RECEIVER_REGISTER_R								: integer := 16#02#;
		-- status of GX receivers
		subtype FREQLOCKED_RANGE is integer range NGX-1 downto 0;
		-- what is displayed is selected by GX_REGISTER_MUX_RANGE
-- below: normal telescope
		subtype SYNCSTATUS_RANGE is integer range NGX-1+NLVDS downto NLVDS;
-- below: FEB		
--		subtype SYNCSTATUS_RANGE is integer range 3*NGX-1 downto 2*NGX;
		-- what is displayed is selected by GX_REGISTER_MUX_RANGE		

		
	constant RECEIVER_RUNTIME_REGISTER_R					: integer := 16#03#;	-- for both LVDS and GX
		-- what is displayed is selected by GX_REGISTER_MUX_RANGE
		-- for "00":
		-- 0x3 A front CH0 	0x4 A back CH0 	0x5 B front CH0 	0x6 B back CH0 
		-- for "01":
		-- 0x3 A front CH1 	0x4 A back CH1 	0x5 B front CH1 	0x6 B back CH1
		-- ...
	constant RECEIVER_ERRCOUNT_REGISTER_R					: integer := 16#07#; -- for both LVDS and GX
		-- what is displayed is selected by GX_REGISTER_MUX_RANGE
		-- for "00":
		-- 0x7 A front CH0 	0x8 A back CH0 	0x9 B front CH0 	0xA B back CH0 
		-- for "01":
		-- 0x7 A front CH1 	0x8 A back CH1 	0x9 B front CH1 	0xA B back CH1
		-- ...
	constant RECEIVER_UNPACKERRCOUNT_REGISTER_R			: integer := 16#0B#;	-- for LVDS, GX and datafaker!
		-- what is displayed is selected by GX_REGISTER_MUX_RANGE
		-- for "00":
		-- 0xB A front CH0 	0xC A back CH0 	0xD B front CH0 	0xE B back CH0 
		-- for "01":
		-- 0xB A front CH1 	0xC A back CH1 	0xD B front CH1 	0xE B back CH1
		-- ...

	constant RECEIVER_UNPACKREADYCOUNT_REGISTER_R			: integer := 16#0C#;	-- for LVDS, GX and datafaker!
		-- to be defined!!!
		-- what is displayed is selected by GX_REGISTER_MUX_RANGE
		-- for "00":
		-- 0xB A front CH0 	0xC A back CH0 	0xD B front CH0 	0xE B back CH0 
		-- for "01":
		-- 0xB A front CH1 	0xC A back CH1 	0xD B front CH1 	0xE B back CH1
		-- ...
		
	constant GENERAL_DEBUG_REGISTER_R						: integer := 16#0F#;
	-- currently used to display 125 MHz counter bits 31 downto 0

	constant LVDS_REGISTER_R									: integer := 16#10#;
		subtype DPALOCKED_RANGE is integer range NLVDS-1 downto 0;
		subtype RXREADY_RANGE is integer range 2*NLVDS-1 downto NLVDS;
	constant LVDS_REGISTER2_R									: integer := 16#11#;
		subtype LVDS_PLL_LOCKED_RANGE is integer range 1 downto 0;
		
	constant INJECTION_COUNTER_A_FRONT_REGISTER_R		: integer := 16#12#;
	-- displays number of injection this entity has delivered
	constant INJECTION_COUNTER_A_BACK_REGISTER_R			: integer := 16#13#;
	constant INJECTION_COUNTER_B_FRONT_REGISTER_R		: integer := 16#14#;
	constant INJECTION_COUNTER_B_BACK_REGISTER_R			: integer := 16#15#;

	constant SLOW_CONTROL_REGISTER_R							: integer := 16#16#;
	-- displays the state of the slow control output signal by the MuPix
		constant SLOW_CONTROL_DOUT_A_FRONT_BIT	: integer := 0;
		constant SLOW_CONTROL_DOUT_A_BACK_BIT	: integer := 1;
		constant SLOW_CONTROL_DOUT_B_FRONT_BIT	: integer := 2;
		constant SLOW_CONTROL_DOUT_B_BACK_BIT	: integer := 3;

	constant MULTICHIP_RO_OVERFLOW_REGISTER_R				: integer := 16#17#;
	-- some debug for Multlink RO

	constant RECEIVER_PRESORT_TOP_REGISTER_R			: integer := 16#18#;
	-- # of Hits presorted from link selected through (DEBUG_CHIP_SELECT_REGISTER_W)(CHIP_SELECT_RANGE): bits 63 downto 32
	constant RECEIVER_PRESORT_BOTTOM_REGISTER_R		: integer := 16#19#;
	-- bits 31 downto 0
	constant RECEIVER_POSTSORT_TOP_REGISTER_R			: integer := 16#1A#;
	-- same after sorting
	constant RECEIVER_POSTSORT_BOTTOM_REGISTER_R		: integer := 16#1B#;	
	
	
	constant RECEIVER_UNPACKERROR_REGISTER_R				: integer := 16#1C#;
	-- error output showing which error occurred in unpacker
	constant TRIGGER_COUNTER_REGISTER_R						: integer := 16#1D#;	--formerly 0x21;
	-- single register showing the counter of the selected trigger 
	constant TRIGGER_SCALER_REGISTER_R						: integer := 16#1E#;	--formerly 0x23;
	-- number of all triggers	
	constant RAW_TRIGGER_COUNTER_REGISTER_R				: integer := 16#1F#;	--formerly 0x22;
	-- raw triggers		
			
	constant MEM_ADDRESS_REGISTER_R 							: integer := 16#20#;
	-- shows the last address in the readmemory that was written
		subtype WRITE_ADDRESS_RANGE is integer range 15 downto 0;		
		subtype EOE_ADDRESS_RANGE is integer range 31 downto 16;
		-- last end of event address
		
	
	-- use eight registers for all hitsorter debug things with select	
--	constant HITSORTER_ERRCOUNT_DEBUG_REGISTER_R			: integer := 16#21#;	--16#1E#; -- mupix7	-- clockout125	
	-- 8 general purpose error register for hitsorter
	-- @ AK: details?
--	constant HITSORTER_ERRCOUNT_IGNOREDHITS_REGISTER_R : integer := 16#0F#;		
--	constant HITSORTER_ERRCOUNT_IGNOREDBLOCKS_REGISTER_R	: integer := 16#1E#;	
--	constant HITSORTER_ERRCOUNT_RESET_REGISTER_R		: integer := 16#1F#;
--	constant HITSORTER_ERRCOUNT_READOLD_REGISTER_R			: integer := 16#2F#;
--	constant HITSORTER_ERRCOUNT_READWRITE_REGISTER_R		: integer := 16#30#;
--	constant HITSORTER_ERRCOUNT_WRITEDONE_REGISTER_R		: integer := 16#31#;
--	constant HITSORTER_ERRCOUNT_ZEROADDR_REGISTER_R		: integer := 16#32#;
--	constant HITSORTER_ERRCOUNT_HCNTEMPTY_REGISTER_R			: integer := 16#33#;	


	constant ADAPTERCARD_DEBUG_REGISTER_R					: integer := 16#21#;	
		-- for RJ45 loopback --> test in/out
		-- write 16#0D# bit 0 --> expect RST_LOCAL_BIT to change
		-- write 16#0D# bit 1 --> expect TRG_LOCAL_BIT to change
		-- depending on SMA loopback:
		-- write 16#01# bit 31 --> expect SPARE_CLK_IN_BIT to change or FPGA_RST_OUT_BIT or FPGA_RST_IN_BIT
		constant RST_LOCAL_BIT 		: integer := 0;
		constant TRG_LOCAL_BIT 		: integer := 1;
		constant SPARE_CLK_IN_BIT 	: integer := 2;
		constant FPGA_RST_OUT_BIT  : integer := 3;
		constant FPGA_RST_IN_BIT   : integer := 4;
		subtype SANTALUZ_SFPPRSNTN_RANGE is integer range 15 downto 8;
		subtype SANTALUZ_RXLOSS_RANGE is integer range 23 downto 16;
		subtype SANTALUZ_TXFAULT_RANGE is integer range 31 downto 24;
		
	
	constant DATA_GEN_REGISTER_R								: integer := 16#22#;	
		-- for debugging of pseudo data generator
		-- state_out = rdcol & ldpix & pri & ldcol & pulldn & mode & rst && readout_state_C & readout_state_B & readout_state_A & 00 & syncres & RO_res
		--             these values are one after the corresponding							readout_state = rdcol ldpix pri ldcol | pulldn rndm syncres rst | rd_C | rd_B | rd_A | 00 syncres RO_res
		--             signal has been enabled for the first time
		--																										rd_A = 0x1 : reset
		--																												 0x2 : StateSync
		--																												 0x3 : StateLdCol2
		--																												 0x4 : StateRdCol2
		--																												 0x5 : StateLdPix2 && PriOutFromDet
		--																												 0x6 : StateSendCounter1
		--																												 0x7 : dataoff (send BC)
		-- mode = 00--> identical diagonals (only for debugging), 
		--			 01--> different diagonals for A,B,C, 
		--			 10--> random (only in external emulator)
		--			 11--> random + something

	
--	constant SLOWCONTROL_READBACK_A_REGISTER_R				: integer := 16#24#; 
		-- displays what's written into the register
		-- NOTE: only the lowest 32 of 3000 bits --> maybe I should write all these into a file?
		
	constant HITSORTER_RECEIVED_HITS_REGISTER_R			: integer	:= 16#23#;
	constant HITSORTER_OUTOFTIME_HITS_REGISTER_R			: integer	:= 16#24#;
	constant HITSORTER_INTIME_HITS_REGISTER_R				: integer	:= 16#25#;
	constant HITSORTER_OVERFLOW_HITS_REGISTER_R			: integer	:= 16#26#;
	constant HITSORTER_MEMWRITE_HITS_REGISTER_R			: integer	:= 16#27#;
	constant HITSORTER_SENT_HITS_REGISTER_R				: integer	:= 16#28#;
	constant HITSORTER_BREAKCOUNTER_REGISTER_R			: integer	:= 16#29#;


	
--	constant RECEIVER_SYNCLOST_SYNCFIFO_REGISTER_R		: integer := 16#2B#;
	-- from here on: 2C, 2D, 2E for other chips
	
	constant DDR3_WR_ADDRESS_REGISTER_R							: integer := 16#2A#;
		subtype DDR3_WR_ADDRESS_RANGE is integer range 23 downto 0;
		-- DDR3 last address written to
		subtype DDR3_STATUS_RANGE				is integer range 31 downto 24;
		-- State of DDR3 interface
	constant DDR3_RD_ADDRESS_REGISTER_R							: integer := 16#2B#;
		subtype DDR3_RD_ADDRESS_RANGE is integer range 23 downto 0;
		-- DDR3 last address that was read
		constant DDR3_LOCAL_INIT_DONE_BIT					:	integer := 30;
		constant DDR3_LOCAL_CAL_SUCCESS_BIT					:	integer := 29;
		constant DDR3_LOCAL_CAL_FAIL_BIT						:	integer := 28;	

	constant TLU_TESTOUT_LSB_REGISTER_R							: integer := 16#2C#; 
	-- TLU stuff
	constant TLU_TESTOUT_MSB_REGISTER_R							: integer := 16#2D#; 
	-- some more TLU stuff
	
	constant HISTOGRAM_1_REGISTER_R								: integer := 16#2E#;
	-- Histogram data #1
	constant HISTOGRAM_2_REGISTER_R								: integer := 16#2F#;	
	-- Histogram data #2	
	
--	constant SYNCFIFO_USEDW_REGISTER_R							: integer := 16#2F#;	-- mupix7 -- clockout125	

	constant DAC_ADC_A_FRONT_REGISTER_R						: integer := 16#30#;
	-- display selected by (DAC_WRITE_REGISTER_W)(DAC_READ_SELECT_RANGE) 
	-- default "00" is ADC data!
	-- selects A_front, C_front,
	constant DAC_ADC_A_BACK_REGISTER_R						: integer := 16#31#;
	constant DAC_ADC_B_FRONT_REGISTER_R						: integer := 16#32#;
	-- selects B_front, E_front,
	constant DAC_ADC_B_BACK_REGISTER_R						: integer := 16#33#;	
	
	constant NIOS_RDDATA_REGISTER_R							: integer := 16#34#;	
	
	constant SLOW_CONTROL_FPGA_1_REGISTER_R				: integer := 16#35#;
		constant DONE_BIT : integer := 0;
		subtype MP8_SC_STATE_RANGE is integer range 31 downto 4;
		
	constant SLOW_CONTROL_FPGA_2_REGISTER_R				: integer := 16#36#;
	
	constant WRITEMEM_READADDR_REGISTER_R					: integer := 16#37#;
		subtype ADDR_RANGE is integer range 15 downto 0;
--	constant RECEIVER_SYNCFIFO_USEDW_R						: integer := 16#37#;
--		constant A_FRONT_LSB : integer := 0;
--		constant A_FRONT_MSB : integer := 2;		
--		subtype A_BACK is integer range 17 downto 15;
--		subtype B_FRONT is integer range 21 downto 19;
--		subtype B_BACK is integer range 25 downto 23;		
	
	-- Registers above 0x38 are in use for the PCIe controller/DMA
	constant DMA_STATUS_REGISTER_R								: integer := 16#38#;
	constant DMA_DATA_ADDR_LOW_REGISTER_R						: integer := 16#39#;
	constant DMA_DATA_ADDR_HI_REGISTER_R						: integer := 16#3A#;
	constant DMA_NUM_PAGES_REGISTER_R							: integer := 16#3B#;  -- bits above 11 not used so far

		
end package mupix_registers;
