-- Register Map

library ieee;
use ieee.std_logic_1164.all;
--use work.mudaq_constants.all;

package mudaq_registers is

		constant LED_REGISTER_W									:  integer := 16#00#;
		constant RESET_REGISTER_W								:  integer := 16#01#;
		constant RESET_BIT_ALL									:  integer := 0;
		constant RESET_BIT_DATAGEN								:  integer := 1;
		constant RESET_BIT_BOARD								:  integer := 2;
		constant RESET_BIT_WORDALIGN							:  integer := 3;
		constant RESET_BIT_RECEIVER							:  integer := 4;
		constant RESET_BIT_DATAFIFO							:  integer := 5;
		constant RESET_BIT_FIFOPLL								:  integer := 6;
		constant RESET_BIT_SC_SLAVE							:  integer := 7;
		constant RESET_BIT_SC_MASTER							:  integer := 8;
		constant RESET_BIT_PCIE_LOCAL							:  integer := 9;
		constant RESET_BIT_TOP_PROC							:  integer := 10;
		constant RESET_BIT_PCIE_APPl							:  integer := 12;
		constant RESET_BIT_EVENT_COUNTER						:  integer := 13;
		constant RESET_BIT_DMA_EVAL							:  integer := 14;
		constant RESET_BIT_LINK_TEST							:  integer := 15;
		constant RESET_BIT_PCIE									:  integer := 31;

		constant DATAGENERATOR_REGISTER_W					: integer := 16#02#;
		constant DATAGENERATOR_BIT_ENABLE					: integer := 0;
		constant DATAGENERATOR_BIT_ENABLE_PIXEL			: integer := 1;
		constant DATAGENERATOR_BIT_ENABLE_FIBRE			: integer := 2;
		constant DATAGENERATOR_BIT_ENABLE_TILE				: integer := 3;
		constant DATAGENERATOR_BIT_ENABLE_TEST				: integer := 4;
		constant DATAGENERATOR_BIT_DMA_HALFFUL_MODE			: integer := 5;
		subtype DATAGENERATOR_FRACCOUNT_RANGE 				is integer range 15 downto 8;
		subtype DATAGENERATOR_NPIXEL_RANGE 					is integer range 15 downto 8;
		subtype DATAGENERATOR_NFIBRE_RANGE 					is integer range 23 downto 16;
		subtype DATAGENERATOR_NTILE_RANGE 					is integer range 31 downto 24;

		constant DATAGENERATOR_DIVIDER_REGISTER_W			: integer := 16#03#;
		
		constant KWORD_W											: integer := 16#04#;

		constant DMA_CONTROL_W								: integer := 16#05#;
			subtype DMA_CONTROL_COUNTER_RANGE 				is integer range 15 downto 0;

		constant DMA_SLOW_DOWN_REGISTER_W					: integer := 16#06#;
		
		constant LINK_TEST_REGISTER_W							: integer := 16#07#;
		constant LINK_TEST_BIT_ENABLE							: integer := 0;

		-- Registers above 0x36 are in use for the PCIe controller/DMA
		constant DMA2_CTRL_ADDR_LOW_REGISTER_W				: integer := 16#36#;
		constant DMA2_CTRL_ADDR_HI_REGISTER_W				: integer := 16#37#;
		constant DMA_REGISTER_W									: integer := 16#38#;
		constant DMA_BIT_ENABLE                      	: integer := 0;
		constant DMA_BIT_NOW                         	: integer := 1;
		constant DMA_BIT_ADDR_WRITE_ENABLE					: integer := 2;
		constant DMA_BIT_ENABLE_INTERRUPTS					: integer := 3;
		constant DMA2_BIT_ENABLE                     	: integer := 16;
		constant DMA2_BIT_NOW                        	: integer := 17;
		constant DMA2_BIT_ADDR_WRITE_ENABLE					: integer := 18;
		constant DMA2_BIT_ENABLE_INTERRUPTS					: integer := 19;

		constant DMA_CTRL_ADDR_LOW_REGISTER_W				: integer := 16#39#;
		constant DMA_CTRL_ADDR_HI_REGISTER_W				: integer := 16#3A#;
		constant DMA_DATA_ADDR_LOW_REGISTER_W				: integer := 16#3B#;
		constant DMA_DATA_ADDR_HI_REGISTER_W				: integer := 16#3C#;
		constant DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W	: integer := 16#3D#;
		subtype DMA_RAM_LOCATION_RANGE 						is integer range 31 downto 20;
		subtype DMA_NUM_PAGES_RANGE							is integer range 19 downto 0;
		constant DMA_NUM_ADDRESSES_REGISTER_W				: integer := 16#3E#;
		subtype DMA_NUM_ADDRESSES_RANGE 						is integer range 11 downto 0; 
		subtype DMA2_NUM_ADDRESSES_RANGE 					is integer range 27 downto 16; 

		--- Read register Map	
		constant	PLL_REGISTER_R									: integer := 16#00#;
		subtype  DIPSWITCH_RANGE 								is integer range 1 downto 0;
		constant	VERSION_REGISTER_R							: integer := 16#01#;
		subtype  VERSION_RANGE 									is integer range 27 downto 0;
		constant EVENTCOUNTER_REGISTER_R						: integer := 16#02#;
		constant EVENTCOUNTER64_REGISTER_R					: integer := 16#03#;
		constant TIMECOUNTER_LOW_REGISTER_R					: integer := 16#04#;
		constant TIMECOUNTER_HIGH_REGISTER_R				: integer := 16#05#;
		constant MEM_WRITEADDR_LOW_REGISTER_R				: integer := 16#06#;		
		constant MEM_WRITEADDR_HIGH_REGISTER_R				: integer := 16#07#;	
		constant EVENT2COUNTER64_REGISTER_R					: integer := 16#08#;
		constant inaddr32_r										: integer := 16#09#;
		constant inaddr32_w										: integer := 16#10#;
		constant DMA_STATUS_R									: integer := 16#11#;
		constant DMA_DATA_WEN									: integer:= 0;
		constant DMA_CONTROL_WEN								: integer:= 1;
		constant PLL_LOCKED_BIT									: integer := 16#12#;
		constant DEBUG_SC											: integer := 16#13#;
		constant DMA_HALFFUL_REGISTER_R						: integer := 16#14#;
		constant DMA_NOTHALFFUL_REGISTER_R					: integer := 16#15#;
		constant DMA_ENDEVENT_REGISTER_R						: integer := 16#16#;
		constant DMA_NOTENDEVENT_REGISTER_R					: integer := 16#17#;
        constant FEBSTATUS_REGISTER_R                       : integer := 16#18#;

		-- Registers above 0x38 are in use for the PCIe controller/DMA
		constant DMA_STATUS_REGISTER_R						: integer := 16#38#;
		constant DMA_DATA_ADDR_LOW_REGISTER_R				: integer := 16#39#;
		constant DMA_DATA_ADDR_HI_REGISTER_R				: integer := 16#3A#;
		constant DMA_NUM_PAGES_REGISTER_R					: integer := 16#3B#;
		constant DMA2_STATUS_REGISTER_R						: integer := 16#3C#;     
		constant DMA2_DATA_ADDR_LOW_REGISTER_R				: integer := 16#3D#;
		constant DMA2_DATA_ADDR_HI_REGISTER_R				: integer := 16#3E#;
		constant DMA2_NUM_PAGES_REGISTER_R					: integer := 16#3F#;
		
end package mudaq_registers;
