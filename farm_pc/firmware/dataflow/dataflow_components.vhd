library ieee;
use ieee.std_logic_1164.all;
use work.pcie_components.all;

package dataflow_components is


	subtype tsrange_type is std_logic_vector(15 downto 0);
	subtype tsupper is range 31 downto 16;
	subtype tslower is range 15 downto 0;
	
	constant tsone is tsrange_type := (others => '1');
	constant tszero is tsrange_type := (others => '1');

	subtype dataplusts_type is std_logic_vector(271 downto 0);
	
	
	component dflowfifo is
      port (
                        data    : in  std_logic_vector(271 downto 0) := (others => 'X'); -- datain
                        wrreq   : in  std_logic                      := 'X';             -- wrreq
                        rdreq   : in  std_logic                      := 'X';             -- rdreq
                        wrclk   : in  std_logic                      := 'X';             -- wrclk
                        rdclk   : in  std_logic                      := 'X';             -- rdclk
                        q       : out std_logic_vector(271 downto 0);                    -- dataout
                        wrusedw : out std_logic_vector(7 downto 0);                      -- wrusedw
                        rdempty : out std_logic;                                         -- rdempty
                        wrfull  : out std_logic                                          -- wrfull
                );
   end component dflowfifo;

	
	component tagram is
                port (
                        data    : in  std_logic_vector(25 downto 0) := (others => 'X'); -- datain
                        address : in  std_logic_vector(15 downto 0) := (others => 'X'); -- address
                        wren    : in  std_logic                     := 'X';             -- wren
                        clock   : in  std_logic                     := 'X';             -- clk
                        q       : out std_logic_vector(25 downto 0)                     -- dataout
                );
        end component tagram;
end package dataflow_components;