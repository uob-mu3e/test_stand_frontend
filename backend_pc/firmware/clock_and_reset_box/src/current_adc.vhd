----------------------------------------------------------------------------------
-- Company: UCL
-- Engineer: Samer Kilani
-- 
-- Create Date: 18.06.2019 12:30:43
-- Design Name: 
-- Module Name: current_adc - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity current_adc is
    Port (
           ser_i_clk : in STD_LOGIC;
           ser_o_clk : out STD_LOGIC;
           cs: out std_logic;
           rd : in STD_LOGIC;
           ser_in: in STD_LOGIC;
           reg_out : out STD_LOGIC_VECTOR (15 downto 0));
end current_adc;

architecture Behavioral of current_adc is
signal adc_counter: unsigned (7 downto 0);
signal tmp: std_logic_vector(15 downto 0);
begin


process (ser_i_clk)
begin
   if rising_edge(ser_i_clk) then
      if (rd='1' and adc_counter = X"2f") then
          adc_counter <= X"00";
      elsif adc_counter /= x"2f" then
          adc_counter <= adc_counter + 1;
    end if;  
   end if;
   
end process;


process (ser_i_clk)
begin
   if rising_edge(ser_i_clk) then
        if (rd='1' and adc_counter = X"2f") then
               tmp <= (others=>'0');
        elsif (adc_counter < X"1f" and adc_counter(0)='1') then
          for i in 0 to 14 loop
              tmp(i+1) <= tmp(i);
          end loop;
          tmp(0) <= ser_in;
        elsif adc_counter = X"1f" then
           reg_out <= tmp;
        end if;
    end if;
end process;

    process (ser_i_clk)
    begin
        if (falling_edge(ser_i_clk)) then
            --if adc_counter < x"1f" then
                ser_o_clk <= adc_counter(0);
            --else
            --    ser_o_clk <= '1';
            --end if;
        end if;
    end process;

--ser_o_clk <= adc_counter(1) when adc_counter < x"1f" else '1';
cs <= '0' when adc_counter < X"20" else '1';
--reg_out(15 downto 8) <= X"00";
--reg_out(7 downto 0) <= std_logic_vector(adc_counter);

end Behavioral;
