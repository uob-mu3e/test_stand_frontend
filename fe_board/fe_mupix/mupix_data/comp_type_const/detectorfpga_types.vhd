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
use work.detectorfpga_constants.all;

package detectorfpga_types is

subtype hit_t is std_logic_vector(HITSIZE-1 downto 0);

type wide_hit_array is array (NINPUTS-1 downto 0) of hit_t;
type hit_array is array (NCHIPS-1 downto 0) of hit_t;

subtype byte_t is std_logic_vector(7 downto 0);
type inbyte_array is array (NINPUTS-1 downto 0) of byte_t;

type state_type is (INIT, START, PRECOUNT, COUNT);

end package detectorfpga_types;