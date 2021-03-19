library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


entity tb_pcie_register_mapping is 
end entity tb_pcie_register_mapping;


architecture TB of tb_pcie_register_mapping is

    constant dataclk_period : time := 4 ns;
    constant pcieclk_period : time := 2 ns;
    
    signal dataclk : std_logic;
    signal pcieclk : std_logic;

    signal rregs_156 : work.util.slv32_array_t(63 downto 0);

begin

	--dataclk
	process begin
		dataclk <= '0';
		wait for dataclk_period/2;
		dataclk <= '1';
		wait for dataclk_period/2;
	end process;
	
	--pcieclk
	process begin
		pcieclk <= '0';
		wait for pcieclk_period/2;
		pcieclk <= '1';
		wait for pcieclk_period/2;
	end process;
	
	rregs_156(1) <= x"FFFFFFFF";
	rregs_156(2) <= x"BBBBBBBB";

    e_mapping : entity work.pcie_register_mapping
        port map(
            --! register inputs for pcie0
            i_pcie0_rregs_156   => rregs_156,
            i_pcie0_rregs_250   => (others => (others => '0')),

            --! register inputs for pcie1
            i_pcie1_rregs_156  => (others => (others => '0')),
            i_pcie1_rregs_250  => (others => (others => '0')),
    
            --! register inputs for pcie0 from a10_block
            i_local_pcie0_rregs_156 => (others => (others => '0')),
            i_local_pcie0_rregs_250 => (others => (others => '0')),

            --! register outputs for pcie0/1
            o_pcie0_rregs       => open,
            o_pcie1_rregs       => open,

            -- slow 156 MHz clock
            i_clk_156           => dataclk,

            -- fast 250 MHz clock
            i_pcie0_clk         => pcieclk,
            i_pcie1_clk         => pcieclk--,
    );
    
end TB;


