-- register map for a10

library ieee;
use ieee.std_logic_1164.all;

package a10_pcie_registers is

		constant LED_REGISTER_W									:  integer := 16#00#;
		constant RESET_REGISTER_W								:  integer := 16#01#;
		constant RESET_BIT_ALL									:  integer := 0;
		constant RESET_BIT_DATAGEN								:  integer := 1;
		constant RESET_BIT_BOARD								:  integer := 2;
		constant RESET_BIT_WORDALIGN							:  integer := 3;
		constant RESET_BIT_RECEIVER							    :  integer := 4;
		constant RESET_BIT_DATAFIFO							    :  integer := 5;
		constant RESET_BIT_FIFOPLL								:  integer := 6;
		constant RESET_BIT_SC_SECONDARY						    :  integer := 7;
		constant RESET_BIT_SC_MAIN    							:  integer := 8;
		constant RESET_BIT_PCIE_LOCAL							:  integer := 9;
		constant RESET_BIT_TOP_PROC							    :  integer := 10;
		constant RESET_BIT_PCIE_APPl							:  integer := 12;
		constant RESET_BIT_EVENT_COUNTER						:  integer := 13;
		constant RESET_BIT_DMA_EVAL							    :  integer := 14;
		constant RESET_BIT_LINK_TEST							:  integer := 15;
        constant RESET_BIT_RUN_START_ACK                        :  integer := 16;
        constant RESET_BIT_RUN_END_ACK                          :  integer := 17;
        constant RESET_BIT_NIOS									:  integer := 18;
		constant RESET_BIT_DDR3									:  integer := 19;
		constant RESET_BIT_DATAFLOW							    :  integer := 20;
        constant RESET_BIT_LINK_MERGER							:  integer := 21;
		constant RESET_BIT_PCIE									:  integer := 31;

		constant DATAGENERATOR_REGISTER_W					    : integer := 16#02#;
		constant DATAGENERATOR_BIT_ENABLE					    : integer := 0;
		constant DATAGENERATOR_BIT_ENABLE_PIXEL			        : integer := 1;
		constant DATAGENERATOR_BIT_ENABLE_FIBRE			        : integer := 2;
		constant DATAGENERATOR_BIT_ENABLE_TILE				    : integer := 3;
		constant DATAGENERATOR_BIT_ENABLE_TEST				    : integer := 4;
		constant DATAGENERATOR_BIT_DMA_HALFFUL_MODE			    : integer := 5;
		    subtype DATAGENERATOR_FRACCOUNT_RANGE 				    is integer range 15 downto 8;
		    subtype DATAGENERATOR_NPIXEL_RANGE 					    is integer range 15 downto 8;
		    subtype DATAGENERATOR_NFIBRE_RANGE 					    is integer range 23 downto 16;
		    subtype DATAGENERATOR_NTILE_RANGE 					    is integer range 31 downto 24;

		constant DATAGENERATOR_DIVIDER_REGISTER_W			    : integer := 16#03#;
		constant KWORD_W										: integer := 16#04#;
		constant DDR3_CONTROL_W									: integer := 16#20#;
		constant DDR3_BIT_ENABLE_A								: integer := 0;
		constant DDR3_BIT_COUNTERTEST_A						    : integer := 1;
		    subtype  DDR3_COUNTERSEL_RANGE_A						is integer range 15 downto 14;		
		constant DDR3_BIT_ENABLE_B								: integer := 16;
		constant DDR3_BIT_COUNTERTEST_B						    : integer := 17;
		    subtype  DDR3_COUNTERSEL_RANGE_B						is integer range 31 downto 30;
		
		constant	DATA_REQ_A_W								: integer := 16#21#;
		constant	DATA_REQ_B_W								: integer := 16#22#;
		constant DATA_TSBLOCK_DONE_W							: integer := 16#23#;
		
        constant DMA_CONTROL_W								    : integer := 16#05#;
			subtype DMA_CONTROL_COUNTER_RANGE 				        is integer range 15 downto 0;
		constant DMA_SLOW_DOWN_REGISTER_W					    : integer := 16#06#;
		
		constant LINK_TEST_REGISTER_W							: integer := 16#07#;
		constant LINK_TEST_BIT_ENABLE							: integer := 0;
        
		constant RUN_NR_REGISTER_W							    : integer := 16#08#;
		constant RUN_NR_ADDR_REGISTER_W							: integer := 16#09#;
		constant FEB_ENABLE_REGISTER_W							: integer := 16#0A#;
		constant DATA_LINK_MASK_REGISTER_W                      : integer := 16#0B#;
		constant GET_N_DMA_WORDS_REGISTER_W                     : integer := 16#0C#;
        constant SC_MAIN_ENABLE_REGISTER_W  				    : integer := 16#0D#;
        constant SC_MAIN_LENGTH_REGISTER_W  				    : integer := 16#0E#;
        constant DATA_LINK_MASK_REGISTER_2_W                      : integer := 16#0F#;

		-- Registers above 0x36 are in use for the PCIe controller/DMA
		constant DMA2_CTRL_ADDR_LOW_REGISTER_W				    : integer := 16#36#;
		constant DMA2_CTRL_ADDR_HI_REGISTER_W				    : integer := 16#37#;
		constant DMA_REGISTER_W									: integer := 16#38#;
		constant DMA_BIT_ENABLE                      	        : integer := 0;
		constant DMA_BIT_NOW                         	        : integer := 1;
		constant DMA_BIT_ADDR_WRITE_ENABLE					    : integer := 2;
		constant DMA_BIT_ENABLE_INTERRUPTS					    : integer := 3;
		constant DMA2_BIT_ENABLE                     	        : integer := 16;
		constant DMA2_BIT_NOW                        	        : integer := 17;
		constant DMA2_BIT_ADDR_WRITE_ENABLE					    : integer := 18;
		constant DMA2_BIT_ENABLE_INTERRUPTS					    : integer := 19;

		constant DMA_CTRL_ADDR_LOW_REGISTER_W				    : integer := 16#39#;
		constant DMA_CTRL_ADDR_HI_REGISTER_W				    : integer := 16#3A#;
		constant DMA_DATA_ADDR_LOW_REGISTER_W				    : integer := 16#3B#;
		constant DMA_DATA_ADDR_HI_REGISTER_W				    : integer := 16#3C#;
		constant DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W	        : integer := 16#3D#;
		    subtype DMA_RAM_LOCATION_RANGE 						    is integer range 31 downto 20;
		    subtype DMA_NUM_PAGES_RANGE							    is integer range 19 downto 0;
		constant DMA_NUM_ADDRESSES_REGISTER_W				    : integer := 16#3E#;
		    subtype DMA_NUM_ADDRESSES_RANGE 						is integer range 11 downto 0; 
		    subtype DMA2_NUM_ADDRESSES_RANGE 					    is integer range 27 downto 16; 

		--- Read register Map	
		constant PLL_REGISTER_R									: integer := 16#00#;
		    subtype  DIPSWITCH_RANGE 								is integer range 1 downto 0;
		constant VERSION_REGISTER_R							    : integer := 16#01#;
		    subtype  VERSION_RANGE 								    is integer range 27 downto 0;
		constant EVENTCOUNTER_REGISTER_R						: integer := 16#02#;
		constant EVENTCOUNTER64_REGISTER_R					    : integer := 16#03#;
		constant TIMECOUNTER_LOW_REGISTER_R					    : integer := 16#04#;
		constant TIMECOUNTER_HIGH_REGISTER_R				    : integer := 16#05#;
		constant MEM_WRITEADDR_LOW_REGISTER_R				    : integer := 16#06#;		
		constant MEM_WRITEADDR_HIGH_REGISTER_R				    : integer := 16#07#;	
		constant EVENT2COUNTER64_REGISTER_R					    : integer := 16#08#;
		constant inaddr32_r										: integer := 16#09#;
		constant inaddr32_w										: integer := 16#10#;
		constant DMA_STATUS_R									: integer := 16#11#;
		constant DMA_DATA_WEN									: integer:= 0;
		constant DMA_CONTROL_WEN								: integer:= 1;
		constant PLL_LOCKED_BIT									: integer := 16#12#;
		constant DEBUG_SC										: integer := 16#13#;
		constant DMA_HALFFUL_REGISTER_R						    : integer := 16#14#;
		constant DMA_NOTHALFFUL_REGISTER_R					    : integer := 16#15#;
		constant DMA_ENDEVENT_REGISTER_R						: integer := 16#16#;
		constant DMA_NOTENDEVENT_REGISTER_R					    : integer := 16#17#;
        constant RUN_NR_ACK_REGISTER_R                          : integer := 16#18#;
        constant RUN_NR_REGISTER_R                              : integer := 16#19#;
        constant RUN_STOP_ACK_REGISTER_R                        : integer := 16#1A#;
        constant BUFFER_STATUS_REGISTER_R                       : integer := 16#1B#;
		constant EVENT_BUILD_STATUS_REGISTER_R                  : integer := 16#1C#;
		constant EVENT_BUILD_DONE								: integer:= 0;
		constant CNT_FIFO_ALMOST_FULL_R                         : integer := 16#1D#;
        constant CNT_TAG_FIFO_FULL_R                  : integer := 16#1E#;
        constant CNT_RAM_FULL_R                       : integer := 16#1F#;
        constant CNT_STREAM_FIFO_FULL_R               : integer := 16#20#;
        constant CNT_DC_LINK_FIFO_FULL_R                : integer := 16#21#;
        constant CNT_SKIP_EVENT_LINK_FIFO_R             : integer := 16#22#;
        constant CNT_SKIP_EVENT_DMA_RAM_R               : integer := 16#23#;
        constant CNT_IDLE_NOT_HEADER_R                  : integer := 16#24#;
        constant CNT_FEB_MERGE_TIMEOUT_R                : integer := 16#25#;
		constant DDR3_STATUS_R									: integer := 16#26#;
		constant DDR3_BIT_CAL_SUCCESS							: integer := 0;
		constant DDR3_BIT_CAL_FAIL								: integer := 1;
		constant DDR3_BIT_RESET_N								: integer := 2;
		constant DDR3_BIT_READY									: integer := 3;
		constant DDR3_BIT_TEST_WRITING						    : integer := 4;
		constant DDR3_BIT_TEST_READING						    : integer := 5;
		constant DDR3_BIT_TEST_DONE							    : integer := 6;
		constant DDR3_ERR_R										: integer := 16#27#;
		constant DATA_TSBLOCKS_R								: integer := 16#28#;
        constant SC_MAIN_STATUS_REGISTER_R						: integer := 16#29#;
            constant SC_MAIN_DONE								: integer := 0;
        constant DDR3_CLK_CNT_R                                 : integer := 16#30#;
        constant SC_STATE_REGISTER_R                            : integer := 16#31#;
		
		-- Registers above 0x38 are in use for the PCIe controller/DMA
		constant DMA_STATUS_REGISTER_R						    : integer := 16#38#;
		constant DMA_DATA_ADDR_LOW_REGISTER_R				    : integer := 16#39#;
		constant DMA_DATA_ADDR_HI_REGISTER_R				    : integer := 16#3A#;
		constant DMA_NUM_PAGES_REGISTER_R					    : integer := 16#3B#;
		constant DMA2_STATUS_REGISTER_R						    : integer := 16#3C#;     
		constant DMA2_DATA_ADDR_LOW_REGISTER_R				    : integer := 16#3D#;
		constant DMA2_DATA_ADDR_HI_REGISTER_R				    : integer := 16#3E#;
		constant DMA2_NUM_PAGES_REGISTER_R					    : integer := 16#3F#;
		
end package;
