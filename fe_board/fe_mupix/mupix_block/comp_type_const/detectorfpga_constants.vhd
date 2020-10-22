---------------------------------------
--
-- On detector FPGA for layer 0 - constant library
-- Sebastian Dittmeier, June 2016
-- 
-- dittmeier@physi.uni-heidelberg.de
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;

package detectorfpga_constants is

constant NINPUTS            : integer := 45;
constant NCHIPS             : integer := 15;
constant NLADDERS           : integer := 5;
constant HITSIZE            : integer := 24;
constant COARSECOUNTERSIZE  : integer := 24;

constant ONE_MILLION        : std_logic_vector(19 downto 0) := x"F4240";
constant ONE_MILLION_int    : integer := 1000000;
constant ONE_MILLION32      : std_logic_vector(31 downto 0) := x"000F4240";
constant ONE_THOUSAND_int   : integer := 1000;
constant HUNDRED_MILLION    : std_logic_vector(27 downto 0) := x"5F5E100"; 
constant HUNDRED_MILLION32  : std_logic_vector(31 downto 0) := x"05F5E100";

constant THIRTYNINE_MILLION : std_logic_vector(27 downto 0) := x"25317C0";
--constant FOURTYONE_MILLION : std_logic_vector(27 downto 0) := x"2719C40";

constant EIGHTY_MILLION     : std_logic_vector(27 downto 0) := x"4C4B400";
constant EIGHTY_THOUSAND    : std_logic_vector(19 downto 0) := x"13880";

constant HUNDREDTWENTYFOUR_MILLION : std_logic_vector(27 downto 0) := x"7641700";
--constant HUNDREDTWENTYSIX_MILLION : std_logic_vector(27 downto 0) := x"7829B80";

constant k28_5 : std_logic_vector(7 downto 0) := "10111100";
constant k28_1 : std_logic_vector(7 downto 0) := "00111100";

end package detectorfpga_constants;