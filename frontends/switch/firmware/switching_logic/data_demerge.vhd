-- receiving and demerging data from FEB
-- (switching board side)
--
-- Martin Mueller, December 2018
-- Marius Koeppel, December 2018

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.protocol.all;

ENTITY data_demerge is 
    PORT(
        clk:                    in  std_logic;
        reset:                  in  std_logic;
        pixel_data:             out std_logic_vector(31 downto 0);
        pixel_data_ready:       out std_logic;
        feb_link_in:            in  std_logic_vector(31 downto 0);
        datak_in:               in  std_logic
    );
END ENTITY data_demerge;

architecture rtl of data_demerge is

----------------signals---------------------
    type   data_demerge_state is (idle,receiving_data, receiving_slowcontrol);
    signal demerge_state : data_demerge_state;
    
----------------begin data_demerge------------------------
BEGIN

    process (clk, reset)
    begin
        if (reset = '1') then 
            demerge_state <= idle;
            pixel_data_ready <= '0';
            pixel_data <= (others => '0');
            
        elsif (rising_edge(clk)) then
        
            if(datak_in = '1') then 
                pixel_data_ready<='0';
            else 
            
                case demerge_state is
                    when idle =>
                        if feb_link_in(27 downto 22) = DATA_HEADER_ID then
                            demerge_state <= receiving_data;
                            pixel_data <= feb_link_in;
                            pixel_data_ready <= '1';
                        elsif feb_link_in(27 downto 22) = SC_HEADER_ID then
                            demerge_state <= receiving_slowcontrol;
                            -- do something with sc data  <= feb_link_in;
                            pixel_data_ready <= '0';
                        else -- "headless" data, something is wrong
                            pixel_data_ready <= '0';
                        end if;
                    
                    when receiving_data =>
                        if(feb_link_in (27 downto 22) = SC_HEADER_ID) then 
                            demerge_state <= receiving_slowcontrol;
                            pixel_data_ready <= '0';
                            -- do something with sc data  <= feb_link_in;
                        elsif (feb_link_in (27 downto 22) = RUN_TAIL_HEADER_ID) then 
                            demerge_state <= idle;
                            pixel_data_ready <= '0';
                        else
                            pixel_data <= feb_link_in;
                            pixel_data_ready <= '1';
                        end if;
                        
                    when receiving_slowcontrol =>
                        if(feb_link_in (27 downto 22) = DATA_HEADER_ID) then 
                            demerge_state <= receiving_data;
                            pixel_data<= feb_link_in;
                            pixel_data_ready <= '1';
                        elsif (feb_link_in (27 downto 22) = RUN_TAIL_HEADER_ID) then 
                            demerge_state <= idle;
                            pixel_data_ready <= '0';
                        else
                            pixel_data_ready <= '0';
                            -- do something with sc data  <= feb_link_in;
                        end if;
                end case;
                
            end if;
        end if;
    end process;
END rtl;
