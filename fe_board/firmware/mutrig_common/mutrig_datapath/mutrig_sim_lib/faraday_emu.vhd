library ieee;
use ieee.std_logic_1164.all;
entity YA2GSD is 
	port(
	I : in std_logic;
	O : out std_logic;
	E : in std_logic;
	E2: in std_logic; 
	E4: in std_logic; 
	E8: in std_logic; 
	SR: in std_logic
);
end entity;

architecture rtl of YA2GSD is
begin 
O <= I after 1 ns when E='1' else 'Z';
end architecture;

library ieee;
use ieee.std_logic_1164.all;
entity XMD is 
	port(
	I : in std_logic;
	O : out std_logic;
	PU : in std_logic;
	PD : in std_logic; 
	SMT: in std_logic
);
end entity;

architecture rtl of XMD is
begin
O <= I after 1 ns;
end architecture;

library ieee;
use ieee.std_logic_1164.all;
entity analog_io_stag is port(a : inout std_logic); end entity;

architecture rtl of analog_io_stag is
begin end architecture;

library ieee;
use ieee.std_logic_1164.all;
entity analog_io is port(a : inout std_logic); end entity;

architecture rtl of analog_io is
begin end architecture;

