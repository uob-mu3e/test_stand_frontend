-- register map for a10

library ieee;
use ieee.std_logic_1164.all;

package a10_pcie_registers is

        constant LED_REGISTER_W                                 :  integer := 16#00#; -- DOC: Used to change LEDs on the boards | FARM
        constant RESET_REGISTER_W                               :  integer := 16#01#; -- DOC: Reset Register | ALL
            constant RESET_BIT_ALL                                  :  integer := 0;  -- DOC: Reset bit to reset all | ALL
            constant RESET_BIT_DATAGEN                              :  integer := 1;  -- DOC: Reset bit for the datagenerator which is generating the link data from FEBs | SWB
            constant RESET_BIT_BOARD                                :  integer := 2;  -- DOC: Not used at the moment | ALL
            constant RESET_BIT_WORDALIGN                            :  integer := 3;  -- DOC: Not used at the moment | ALL
            constant RESET_BIT_RECEIVER                             :  integer := 4;  -- DOC: Not used at the moment | ALL
            constant RESET_BIT_DATAFIFO                             :  integer := 5;  -- DOC: Not used at the moment | ALL
            constant RESET_BIT_FIFOPLL                              :  integer := 6;  -- DOC: Not used at the moment | ALL
            constant RESET_BIT_SC_SECONDARY                         :  integer := 7;  -- DOC: Reset bit for the slowcontrol secondary | SWB
            constant RESET_BIT_SC_MAIN                              :  integer := 8;  -- DOC: Reset bit for the slowcontrol main | SWB
            constant RESET_BIT_PCIE_LOCAL                           :  integer := 9;  -- DOC: Not used at the moment | ALL
            constant RESET_BIT_TOP_PROC                             :  integer := 10; -- DOC: Not used at the moment | ALL
            constant RESET_BIT_PCIE_APPl                            :  integer := 12; -- DOC: Not used at the moment | ALL
            constant RESET_BIT_EVENT_COUNTER                        :  integer := 13; -- DOC: Reset bit for the swb_data_demerger | SWB
            constant RESET_BIT_DMA_EVAL                             :  integer := 14; -- DOC: Reset bit for DMA evaluationg / monitoring for PCIe 0 | ALL
            constant RESET_BIT_LINK_TEST                            :  integer := 15; -- DOC: Not used at the moment | ALL
            constant RESET_BIT_RUN_START_ACK                        :  integer := 16; -- DOC: Rest bit for seeing the run ack in run_control | SWB
            constant RESET_BIT_RUN_END_ACK                          :  integer := 17; -- DOC: Rest bit for seeing the run end in run_control | SWB
            constant RESET_BIT_NIOS                                 :  integer := 18; -- DOC: Not used at the moment | ALL
            constant RESET_BIT_DDR3                                 :  integer := 19; -- DOC: Reset bit for DDR3 control entitie | FARM
            constant RESET_BIT_DATAFLOW                             :  integer := 20; -- DOC: Not used at the moment | ALL
            constant RESET_BIT_LINK_MERGER                          :  integer := 21; -- DOC: Not used at the moment | ALL
            constant RESET_BIT_DATA_PATH                            :  integer := 22; -- DOC: Reset bit for the data path | SWB
            constant RESET_BIT_FARM_DATA_PATH                       :  integer := 23; -- DOC: Reset bit for the data path | SWB
            constant RESET_BIT_PCIE                                 :  integer := 31; -- DOC: Not used at the moment | ALL

        constant DATAGENERATOR_REGISTER_W                       : integer := 16#02#; -- DOC: Register to control the datagenerator which is generating the link data from FEBs | SWB
            constant DATAGENERATOR_BIT_ENABLE                       : integer := 0;  -- DOC: Not used at the moment | SWB
            constant DATAGENERATOR_BIT_ENABLE_PIXEL                 : integer := 1;  -- DOC: Bit to enable pixel data | SWB
            constant DATAGENERATOR_BIT_ENABLE_FIBRE                 : integer := 2;  -- DOC: Bit to enable fibre data | SWB
            constant DATAGENERATOR_BIT_ENABLE_TILE                  : integer := 3;  -- DOC: Bit to enable tile data | SWB
            constant DATAGENERATOR_BIT_ENABLE_TEST                  : integer := 4;  -- DOC: Not used at the moment | SWB
            constant DATAGENERATOR_BIT_DMA_HALFFUL_MODE             : integer := 5;  -- DOC: Not used at the moment | SWB
            subtype DATAGENERATOR_FRACCOUNT_RANGE                   is integer range 15 downto 8; -- DOC: Not used at the moment | SWB
            subtype DATAGENERATOR_NPIXEL_RANGE                      is integer range 15 downto 8; -- DOC: Not used at the moment | SWB
            subtype DATAGENERATOR_NFIBRE_RANGE                      is integer range 23 downto 16; -- DOC: Not used at the moment | SWB
            subtype DATAGENERATOR_NTILE_RANGE                       is integer range 31 downto 24; -- DOC: Not used at the moment | SWB

        constant DATAGENERATOR_DIVIDER_REGISTER_W               : integer := 16#03#; -- DOC: Register to slow down the datagenerator which is generating the link data from FEBs | SWB
        constant KWORD_W                                        : integer := 16#04#; -- DOC: Not used at the moment | ALL
        constant DMA_CONTROL_W                                  : integer := 16#05#; -- DOC: Not used at the moment | ALL
            subtype DMA_CONTROL_COUNTER_RANGE                       is integer range 15 downto 0; -- DOC: Not used at the moment | ALL
        constant DMA_SLOW_DOWN_REGISTER_W                       : integer := 16#06#; -- DOC: Not used at the moment | ALL
        
        constant LINK_TEST_REGISTER_W                           : integer := 16#07#; -- DOC: Not used at the moment | ALL
        constant LINK_TEST_BIT_ENABLE                               : integer := 0;  -- DOC: Not used at the moment | ALL
        
        constant RUN_NR_REGISTER_W                              : integer := 16#08#; -- DOC: Register to write the current run number | SWB
        constant RUN_NR_ADDR_REGISTER_W                         : integer := 16#09#; -- DOC: Ask for run number of FEB with this addr | SWB
        constant FEB_ENABLE_REGISTER_W                          : integer := 16#0A#; -- DOC: Enable register for FEBs for run control and slowcontrol | SWB
        constant DATA_LINK_MASK_REGISTER_W                      : integer := 16#0B#; -- DOC: Not used at the moment | SWB
        constant GET_N_DMA_WORDS_REGISTER_W                     : integer := 16#0C#; -- DOC: Number of requested words which will be send via DMA | SWB
        constant SC_MAIN_ENABLE_REGISTER_W                      : integer := 16#0D#; -- DOC: Reg that the slowcontrol main only starts on a 0->1 transition | SWB
        constant SC_MAIN_LENGTH_REGISTER_W                      : integer := 16#0E#; -- DOC: Lenght (15 downto 0) for slowcontrol package | SWB
        constant DATA_LINK_MASK_REGISTER_2_W                    : integer := 16#0F#; -- DOC: Not used at the moment | SWB
        constant SWB_LINK_MASK_PIXEL_REGISTER_W                 : integer := 16#10#; -- DOC: Mask Pixel links | SWB
        constant SWB_LINK_MASK_SCIFI_REGISTER_W                 : integer := 16#11#; -- DOC: Mask Scifi links | SWB
        constant SWB_LINK_MASK_TILES_REGISTER_W                 : integer := 16#12#; -- DOC: Mask Tile links | SWB
        constant SWB_READOUT_STATE_REGISTER_W                   : integer := 16#13#; -- DOC: Readout state | SWB
            constant USE_BIT_GEN_LINK                               : integer := 0;  -- DOC: Readout state where link data is generated (for debugging) | SWB
            constant USE_BIT_STREAM                                 : integer := 1;  -- DOC: Readout state where link data is read out in round-robin | SWB
            constant USE_BIT_MERGER                                 : integer := 2;  -- DOC: Readout state where link data is time aligned (tree) | SWB
            constant USE_BIT_GEN_MERGER                             : integer := 4;  -- DOC: Readout state where the time aligned data is generated (for debugging) | SWB
            constant USE_BIT_FARM                                   : integer := 5;  -- DOC: Readout state where the data is send to the farm (1) else (0) it is readout via DMA on the SWB | SWB
            constant USE_BIT_TEST                                   : integer := 6;  -- DOC: Not used at the moment | SWB
            constant USE_BIT_PIXEL                                  : integer := 7;  -- DOC: Readout state to only readout pixel data via DMA (for debugging) | SWB
            constant USE_BIT_SCIFI                                  : integer := 8;  -- DOC: Readout state to only readout scifi data via DMA (for debugging) | SWB
        constant SWB_READOUT_LINK_REGISTER_W                    : integer := 16#14#; -- DOC: Not used at the moment | SWB
        
        constant SWB_COUNTER_REGISTER_W                         : integer := 16#15#; -- DOC: Register to readout counter values from the SWB, to have more information about the counter look at a10_counter.md | SWB
            subtype SWB_COUNTER_ADDR_RANGE                          is integer range 7 downto 0;    -- DOC: Addr of the counter which should be read out | SWB
            subtype SWB_LINK_RANGE                                  is integer range 15 downto 8;   -- DOC: Link addrs for link specific counters (#events, #subheaders, etc.) | SWB

        constant DDR3_CONTROL_W                                 : integer := 16#20#; -- DOC: Control register for the ddr3_memory_controller | FARM
            constant DDR3_BIT_ENABLE_A                              : integer := 0;  -- DOC: Enable statemachine of DDR-A | FARM
            constant DDR3_BIT_COUNTERTEST_A                         : integer := 1;  -- DOC: Enable counter test (1) or dataflow (0) of DDR-A | FARM
            subtype  DDR3_COUNTERSEL_RANGE_A                        is integer range 15 downto 14; -- DOC: "01" -> get poserr_reg, "10" -> get counterr_reg else cur time counter written to DDR3 | FARM
            constant DDR3_BIT_ENABLE_B                              : integer := 16;  -- DOC: Enable statemachine of DDR-B | FARM
            constant DDR3_BIT_COUNTERTEST_B                         : integer := 17;  -- DOC: Enable counter test (1) or dataflow (0) of DDR-B | FARM
            subtype  DDR3_COUNTERSEL_RANGE_B                        is integer range 31 downto 30; -- DOC: "01" -> get poserr_reg, "10" -> get counterr_reg else cur time counter written to DDR3 | FARM
        
        constant DATA_REQ_A_W                                   : integer := 16#21#; -- DOC: Register for requesting subheaders from DDR3, for SUBTS in GPU_SUBTS_SEL DO writereg(SUBTS) | FARM
        constant DATA_REQ_B_W                                   : integer := 16#22#; -- DOC: Register for requesting subheaders from DDR3, for SUBTS in GPU_SUBTS_SEL DO writereg(SUBTS) | FARM
        constant DATA_TSBLOCK_DONE_W                            : integer := 16#23#; -- DOC: dynamic limit when we change from writing to reading (15 downto 8 from 35 downto 4 of the 48b TS) | FARM
        constant FARM_READOUT_STATE_REGISTER_W                  : integer := 16#24#; -- DOC: Readout state | FARM
            constant USE_BIT_GEN_MERGE                              : integer := 0;  -- DOC: Generate SWB data (for dubugging) | FARM
        constant FARM_ID_REGISTER_W                             : integer := 16#25#; -- DOC: Farm ID written to the reserved filed of the MIDAS bank | FARM
        constant FARM_REQ_EVENTS_W                              : integer := 16#26#; -- DOC: total number of requested events (should match to len(DATA_REQ_A/B_W) from GPU selection | FARM
        constant FARM_CTL_REGISTER_W                            : integer := 16#27#;
            constant USE_BIT_PIXEL_ONLY                             : integer := 0;
            constant USE_BIT_SCIFI_ONLY                             : integer := 1;

        -- Registers above 0x36 are in use for the PCIe controller/DMA
        constant DMA2_CTRL_ADDR_LOW_REGISTER_W                  : integer := 16#36#;
        constant DMA2_CTRL_ADDR_HI_REGISTER_W                   : integer := 16#37#;
        constant DMA_REGISTER_W                                 : integer := 16#38#;
            constant DMA_BIT_ENABLE                             : integer := 0;
            constant DMA_BIT_NOW                                : integer := 1;
            constant DMA_BIT_ADDR_WRITE_ENABLE                  : integer := 2;
            constant DMA_BIT_ENABLE_INTERRUPTS                  : integer := 3;
            constant DMA2_BIT_ENABLE                            : integer := 16;
            constant DMA2_BIT_NOW                               : integer := 17;
            constant DMA2_BIT_ADDR_WRITE_ENABLE                 : integer := 18;
            constant DMA2_BIT_ENABLE_INTERRUPTS                 : integer := 19;

        constant DMA_CTRL_ADDR_LOW_REGISTER_W                   : integer := 16#39#;
        constant DMA_CTRL_ADDR_HI_REGISTER_W                    : integer := 16#3A#;
        constant DMA_DATA_ADDR_LOW_REGISTER_W                   : integer := 16#3B#;
        constant DMA_DATA_ADDR_HI_REGISTER_W                    : integer := 16#3C#;
        constant DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W          : integer := 16#3D#;
            subtype DMA_RAM_LOCATION_RANGE                          is integer range 31 downto 20;
            subtype DMA_NUM_PAGES_RANGE                             is integer range 19 downto 0;
        constant DMA_NUM_ADDRESSES_REGISTER_W                   : integer := 16#3E#;
            subtype DMA_NUM_ADDRESSES_RANGE                         is integer range 11 downto 0; 
            subtype DMA2_NUM_ADDRESSES_RANGE                        is integer range 27 downto 16; 

        --- Read register Map	
        constant PLL_REGISTER_R                                 : integer := 16#00#;
            subtype  DIPSWITCH_RANGE                                is integer range 1 downto 0;
        constant VERSION_REGISTER_R                             : integer := 16#01#;
            subtype  VERSION_RANGE                                  is integer range 27 downto 0;
        constant EVENTCOUNTER_REGISTER_R                        : integer := 16#02#;
        constant EVENTCOUNTER64_REGISTER_R                      : integer := 16#03#;
        constant TIMECOUNTER_LOW_REGISTER_R                     : integer := 16#04#;
        constant TIMECOUNTER_HIGH_REGISTER_R                    : integer := 16#05#;
        constant MEM_WRITEADDR_LOW_REGISTER_R                   : integer := 16#06#;
        constant MEM_WRITEADDR_HIGH_REGISTER_R                  : integer := 16#07#;
        constant EVENT2COUNTER64_REGISTER_R                     : integer := 16#08#;
        constant inaddr32_r                                     : integer := 16#09#;
        constant inaddr32_w                                     : integer := 16#10#;
        constant DMA_STATUS_R                                   : integer := 16#11#;
            constant DMA_DATA_WEN                                   : integer:= 0;
            constant DMA_CONTROL_WEN                                : integer:= 1;
        constant PLL_LOCKED_REGISTER_R                          : integer := 16#12#;
        constant DEBUG_SC                                       : integer := 16#13#;
        constant DMA_HALFFUL_REGISTER_R                         : integer := 16#14#;
        constant DMA_NOTHALFFUL_REGISTER_R                      : integer := 16#15#;
        constant DMA_ENDEVENT_REGISTER_R                        : integer := 16#16#;
        constant DMA_NOTENDEVENT_REGISTER_R                     : integer := 16#17#;
        constant RUN_NR_ACK_REGISTER_R                          : integer := 16#18#;
        constant RUN_NR_REGISTER_R                              : integer := 16#19#;
        constant RUN_STOP_ACK_REGISTER_R                        : integer := 16#1A#;
        constant BUFFER_STATUS_REGISTER_R                       : integer := 16#1B#;
        constant EVENT_BUILD_STATUS_REGISTER_R                  : integer := 16#1C#;
            constant EVENT_BUILD_DONE                               : integer:= 0;
        constant CNT_FIFO_ALMOST_FULL_R                         : integer := 16#1D#;
        constant CNT_TAG_FIFO_FULL_R                            : integer := 16#1E#;
        constant CNT_RAM_FULL_R                                 : integer := 16#1F#;
        constant CNT_STREAM_FIFO_FULL_R                         : integer := 16#20#;
        constant CNT_DC_LINK_FIFO_FULL_R                        : integer := 16#21#;
        constant CNT_SKIP_EVENT_LINK_FIFO_R                     : integer := 16#22#;
        constant CNT_SKIP_EVENT_DMA_RAM_R                       : integer := 16#23#;
        constant CNT_IDLE_NOT_HEADER_R                          : integer := 16#24#;
        constant CNT_FEB_MERGE_TIMEOUT_R                        : integer := 16#25#;
        constant DDR3_STATUS_R                                  : integer := 16#26#;
            constant DDR3_BIT_CAL_SUCCESS                           : integer := 0;
            constant DDR3_BIT_CAL_FAIL                              : integer := 1;
            constant DDR3_BIT_RESET_N                               : integer := 2;
            constant DDR3_BIT_READY                                 : integer := 3;
            constant DDR3_BIT_TEST_WRITING                          : integer := 4;
            constant DDR3_BIT_TEST_READING                          : integer := 5;
            constant DDR3_BIT_TEST_DONE                             : integer := 6;
        constant DDR3_ERR_R                                     : integer := 16#27#;
        constant DATA_TSBLOCKS_R                                : integer := 16#28#;
        constant SC_MAIN_STATUS_REGISTER_R                      : integer := 16#29#;
            constant SC_MAIN_DONE                                   : integer := 0;
        constant DDR3_CLK_CNT_R                                 : integer := 16#30#;
        constant SC_STATE_REGISTER_R                            : integer := 16#31#;
        constant DMA_CNT_WORDS_REGISTER_R                       : integer := 16#32#;
        constant SWB_COUNTER_REGISTER_R                         : integer := 16#33#;
        constant SWB_COUNTER_REGISTER_ADDR_R                    : integer := 16#34#;
        
        -- Registers above 0x38 are in use for the PCIe controller/DMA
        constant DMA_STATUS_REGISTER_R                          : integer := 16#38#;
        constant DMA_DATA_ADDR_LOW_REGISTER_R                   : integer := 16#39#;
        constant DMA_DATA_ADDR_HI_REGISTER_R                    : integer := 16#3A#;
        constant DMA_NUM_PAGES_REGISTER_R                       : integer := 16#3B#;
        constant DMA2_STATUS_REGISTER_R                         : integer := 16#3C#;     
        constant DMA2_DATA_ADDR_LOW_REGISTER_R                  : integer := 16#3D#;
        constant DMA2_DATA_ADDR_HI_REGISTER_R                   : integer := 16#3E#;
        constant DMA2_NUM_PAGES_REGISTER_R                      : integer := 16#3F#;
        
end package;
