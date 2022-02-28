library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.a10_counters.all;
use work.a10_pcie_registers.all;

entity tb_counter_readout is
end entity;


architecture TB of tb_counter_readout is

    signal reset_n  : std_logic;
    signal clk      : std_logic;
    constant dataclk_period : time := 4 ns;

    constant g_NLINKS_DATA_PIXEL : positive := 10;

    signal i_writeregs_250, o_readregs_250 : work.util.slv32_array_t(63 downto 0);

    --! counters
    signal counter_swb_data_pixel_156 : work.util.slv32_array_t(g_NLINKS_DATA_PIXEL*5-1 downto 0);
    signal counter_swb_data_pixel_250 : work.util.slv32_array_t(4 downto 0);

begin

    e_counters : entity work.swb_readout_counters
    generic map (
        g_A_CNT             => 5,
        g_B_CNT             => g_NLINKS_DATA_PIXEL * 5,
        g_NLINKS_DATA_SCIFI => 1,
        g_NLINKS_DATA_PIXEL => g_NLINKS_DATA_PIXEL--,
    )
    port map (
        --! register inputs for pcie0
        i_wregs_add_A       => i_writeregs_250(SWB_COUNTER_REGISTER_W),

        --! counters
        i_counter_A         => counter_swb_data_pixel_250, -- pcie clk
        i_counter_B         => counter_swb_data_pixel_156, -- link clk

        --! register outputs for pcie0
        o_pcie_data         => o_readregs_250(SWB_COUNTER_REGISTER_R),
        o_pcie_addr         => o_readregs_250(SWB_COUNTER_REGISTER_ADDR_R),

        --! i_reset
        i_reset_n_A         => reset_n,

        --! clocks
        i_clk_A             => clk,
        i_clk_B             => clk--,
    );


    gen_counter_156 : FOR i in 0 to g_NLINKS_DATA_PIXEL*5 - 1 GENERATE
        process(clk, reset_n)
        begin
        if ( reset_n = '0' ) then
            counter_swb_data_pixel_156(i) <= (others => '0');
        elsif rising_edge(clk) then
            counter_swb_data_pixel_156(i) <= counter_swb_data_pixel_156(i) + '1';
        end if;
        end process;
    END GENERATE;

    gen_counter_250 : FOR i in 0 to 5 - 1 GENERATE
        process(clk, reset_n)
        begin
        if ( reset_n = '0' ) then
            counter_swb_data_pixel_250(i) <= (others => '0');
        elsif rising_edge(clk) then
            counter_swb_data_pixel_250(i) <= counter_swb_data_pixel_250(i) + '1';
        end if;
        end process;
    END GENERATE;

    -- clk
    process begin
        clk <= '0';
        wait for dataclk_period/2;
        clk <= '1';
        wait for dataclk_period/2;
    end process;

    -- reset_n
    process begin
        reset_n <= '0';
        wait for 20 ns;
        reset_n <= '1';
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000000";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000001";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000002";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000003";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000004";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000005";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000006";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000007";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000008";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000104";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000105";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000106";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000107";
        wait for 20 ns;
        i_writeregs_250(SWB_COUNTER_REGISTER_W) <= x"00000108";
        wait;
    end process;

end architecture;
