---------------------------------------
-- Firefly ECUO 14G x4 - constant library
-- Martin Mueller, August 2020
----------------------------------

library ieee;
use ieee.std_logic_1164.all;

package firefly_constants is

-- device I2c address
constant FFLY_DEV_ADDR_7        : std_logic_vector(6 downto 0) := "1010000";
constant FFLY_DEV_ADDR_8_WRITE  : std_logic_vector(7 downto 0) := x"A0";
constant FFLY_DEV_ADDR_8_READ   : std_logic_vector(7 downto 0) := x"A1";

-- commands
constant CMD_READ           : std_logic_vector(7 downto 0) := "10100001"; -- read current address
constant CMD_WRITE          : std_logic_vector(7 downto 0) := "10100000";

-- mem addr lower Page 00 (incomplete)
constant ADDR_OP_TIME_1     : std_logic_vector(7 downto 0) := "00010011";
constant ADDR_OP_TIME_2     : std_logic_vector(7 downto 0) := "00010100";
constant ADDR_TEMPERATURE   : std_logic_vector(7 downto 0) := "00010110";
constant RX1_PWR_1  : std_logic_vector(7 downto 0) := "00100010";
constant RX1_PWR_2  : std_logic_vector(7 downto 0) := "00100011";

end package firefly_constants;