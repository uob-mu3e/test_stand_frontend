-- Register Map

library ieee;
use ieee.std_logic_1164.all;

package reg_map_s4 is

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
		
	constant INJECTION_DAC_A_FRONT_REGISTER_W	: integer := 16#15#;	
	constant INJECTION_DAC_A_BACK_REGISTER_W 	: integer := 16#18#;
		subtype INJECTION1_RANGE 			is integer range 15 downto 0;
		subtype THRESHOLD_PIX_RANGE 		is integer range 31 downto 16;
		
	constant THRESHOLD_DAC_A_FRONT_REGISTER_W	: integer := 16#11#;
		subtype THRESHOLD_LOW_RANGE 		is integer range 15 downto 0;
		subtype THRESHOLD_HIGH_RANGE 		is integer range 31 downto 16;
		
	constant TEMP_A_FRONT_REGISTER_W	: integer := 16#19#;
		subtype TEMP_DAC_RANGE 				is integer range 15 downto 0;
		subtype TEMP_ADC_W_RANGE 			is integer range 31 downto 16;

		
end package reg_map_s4;
