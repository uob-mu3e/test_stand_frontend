-- Reset MuTRiG
-- sync resets to base clock
-- adabdet from Nik


library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.mudaq_registers.all;

entity mutrig_reset is 
  port(
    clk_pci            : in  std_logic;                       -- fast pcie clock 250MHz
    clk_base           : in  std_logic;                       -- 125MHz system clock, use 625 later?
    clk_slow           : in  std_logic;                       -- slow controle clock
    reset_n            : in  std_logic;                       -- 
    reset_reg          : in  std_logic_vector(31 downto 0);   -- reset register
    reset_reg_written  : in  std_logic;                       -- 
    o_reset_slow       : out std_logic_vector(31 downto 0);   -- resets sync to clk_slow (50MHz)
    o_reset_base       : out std_logic_vector(31 downto 0);   -- resets sync to clk_base (125MHz)
    o_reset_ch         : out std_logic;                       -- active high, (forces cc to 0)
    o_reset_chip       : out std_logic                        -- active high (1 cyle -> reset_ch, 
                                                              -- multiple cycles -> resets state machine/TDC)
  );
end entity mutrig_reset;


architecture rtl of mutrig_reset is


  signal reset_slow, reset_base                                          : std_logic_vector(31 downto 0);
  signal reset_slow0, reset_slow1, reset_slow2, reset_slow3, reset_slow4 : std_logic_vector(31 downto 0);
  
  signal reset_chip_shift : std_logic_vector(31 downto 0);

	
begin
  -- prolong 250MHz reset signals to make sure they are active once in the 50Mhz slow clk
  process(clk_pci, reset_n) 
    begin
      if(reset_n = '0') then -- async active low rst
	  -- reset_slow  <= (others =>'0');
          reset_slow0      <= (others =>'0');
          reset_slow1      <= (others =>'0');
          reset_slow2      <= (others =>'0');
          reset_slow3      <= (others =>'0');
          reset_slow4      <= (others =>'0');
			 reset_chip_shift <= (others =>'0');
      elsif(clk_pci'event and clk_pci = '1') then
          reset_slow <= reset_slow0 or reset_slow1 or reset_slow2 or reset_slow3 or reset_slow4; -- make sure the reset signal is at least one slow clk cycle long
          reset_base <= reset_slow0 or reset_slow1; -- make sure the reset signal is at least one base clk cycle long

          reset_slow4 <= reset_slow3;
          reset_slow3 <= reset_slow2;
          reset_slow2 <= reset_slow1;
          reset_slow1 <= reset_slow0;
          reset_slow0 <= (others =>'0'); -- default off 
			 reset_chip_shift(0) <=  reset_slow4(RESET_BIT_MUTRIG_CHIP);
			 for j in 30 downto 0 loop
			    reset_chip_shift(j+1) <= reset_chip_shift(j); 
			 end loop;

          if(reset_reg_written = '1' and reset_reg(RESET_BIT_ALL) = '1') then
              reset_slow0 <= (others => '1');
	  else
              for i in 31 downto 0 loop
                  if(reset_reg_written = '1' and reset_reg(i) = '1') then
                      reset_slow0(i) <= '1';
                  end if;
              end loop;
	  end if;

          -- special case for hold resets
	  reset_slow0(RESET_BIT_MUTRIG_CHANNEL_HOLD) <= reset_reg(RESET_BIT_MUTRIG_CHANNEL_HOLD);
      end if;
  end process;

  -- sync slow outputs to slow clk
  process(clk_slow)
    begin
        if(clk_slow'event and clk_slow = '1') then
	    o_reset_slow <= reset_slow;
        end if;	    
  end process;

  -- sync base outputs to base clk
  process(clk_base)
    begin
        if(clk_base'event and clk_base = '1') then
            o_reset_base <= reset_base;
            o_reset_ch   <= reset_base(RESET_BIT_MUTRIG_CHANNEL) or reset_base(RESET_BIT_MUTRIG_CHANNEL_HOLD);
            o_reset_chip <= reset_slow(RESET_BIT_MUTRIG_CHIP) or reset_chip_shift( 0) or 
				                                                     reset_chip_shift( 1) or 
																					  reset_chip_shift( 2) or 
																					  reset_chip_shift( 3) or 
																					  reset_chip_shift( 4) or 
																					  reset_chip_shift( 5) or 
																					  reset_chip_shift( 6) or 
																					  reset_chip_shift( 7) or 
																					  reset_chip_shift( 8) or 
																					  reset_chip_shift( 9) or 
																					  reset_chip_shift(10) or 
																					  reset_chip_shift(11) or 
																					  reset_chip_shift(12) or 
																					  reset_chip_shift(13) or 
																					  reset_chip_shift(14) or  
																					  reset_chip_shift(14) or  
																					  reset_chip_shift(15) or  
																					  reset_chip_shift(16) or  
																					  reset_chip_shift(17) or  
																					  reset_chip_shift(18) or  
																					  reset_chip_shift(19) or  
																					  reset_chip_shift(20) or  
																					  reset_chip_shift(21) or  
																					  reset_chip_shift(22) or  
																					  reset_chip_shift(23) or  
																					  reset_chip_shift(24) or  
																					  reset_chip_shift(25) or 
																					  reset_chip_shift(26) or 
																					  reset_chip_shift(27) or 
																					  reset_chip_shift(28) or 
																					  reset_chip_shift(29) or 
																					  reset_chip_shift(30); 
			  	--reset_slow0(RESET_BIT_MUTRIG_CHIP) or 
            --reset_slow1(RESET_BIT_MUTRIG_CHIP) or 
            --reset_slow2(RESET_BIT_MUTRIG_CHIP) or 
            --reset_slow3(RESET_BIT_MUTRIG_CHIP); -- 4 x 4 ns = 2 x base_clk cycles
	                    -- could be operated with chip reset only -> 1 cylce == channel reset:
	                    -- or reset_base(RESET_BIT_MUTRIG_CHANNEL) ;
	end if;
  end process;
end rtl;
