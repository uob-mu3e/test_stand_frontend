-- receive run control signals from FEBs
-- TODO more than one FEB

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

ENTITY run_control is 
    PORT(
		i_clk:                              in  std_logic; -- receive clock (156.25 MHz)
        i_reset_n:                          in  std_logic;
		i_aligned:                          in  std_logic; -- word alignment achieved
		i_data:                             in  std_logic_vector(31 downto 0); -- optical from frontend board
		i_datak:                            in  std_logic_vector(3 downto 0);
        o_FEB_status:                       out std_logic_vector(31 downto 0)
);
END ENTITY run_control;

architecture rtl of run_control is

signal run_prep_acknowledge_received :          std_logic;
signal end_of_run_received :                    std_logic;

BEGIN

    process (i_clk, i_reset_n, i_aligned)
    begin
        if (i_reset_n = '0' or i_aligned = '0') then 
            run_prep_acknowledge_received    <= '0';
            end_of_run_received              <= '0';
            FEB_status                       <= (others => '0');
            
        elsif (rising_edge(clk)) then
            o_FEB_status                    <=
                (0 => run_prep_acknowledge_received,
                 1 => end_of_run_received,
                 others => '0');
                 
            if(i_data = run_prep_acknowledge and i_datak = run_prep_acknowledge_datak) then 
                run_prep_acknowledge_received   <= '1';
                end_of_run_received             <= '0';
                
            elsif (i_data = RUN_END and i_datak = RUN_END_DATAK) then
                run_prep_acknowledge_received   <= '0';
                end_of_run_received             <= '1';
            end if;
		end if;
    end process;
END rtl;