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
constant RX1_PWR1  : std_logic_vector(7 downto 0) := "00100010";
constant RX1_PWR2  : std_logic_vector(7 downto 0) := "00100011";
constant RX2_PWR1  : std_logic_vector(7 downto 0) := "00100100";
constant RX2_PWR2  : std_logic_vector(7 downto 0) := "00100101";
constant RX3_PWR1  : std_logic_vector(7 downto 0) := "00100110";
constant RX3_PWR2  : std_logic_vector(7 downto 0) := "00100111";
constant RX4_PWR1  : std_logic_vector(7 downto 0) := "00101000";
constant RX4_PWR2  : std_logic_vector(7 downto 0) := "00101001";

type RX_PWR_TYPE is array (0 to 7) OF std_logic_vector(7 downto 0);
-- is it possible to somehow use the constants from above here (like :=(RX1_PWR1,RX1_PWR2,...) )? Could not find a way to do that
constant ADDR_RX_PWR        : RX_PWR_TYPE         := ("00100010","00100011","00100100","00100101","00100110","00100111","00101000","00101001");

end package firefly_constants;