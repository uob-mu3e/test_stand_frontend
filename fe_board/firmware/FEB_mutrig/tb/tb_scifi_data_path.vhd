library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;
use work.scifi_registers.all;

entity tb_scifi_data_path is
end entity;

architecture arch of tb_scifi_data_path is

    constant CLK_MHZ    : real := 10000.0; -- MHz
    constant N_LINKS    : integer := 2;
    constant N_ASICS    : integer := 4;
    constant N_MODULES  : integer := 2;

    signal clk, reset_n : std_logic := '0';

    signal scifi_reg        : work.util.rw_t;    
    signal run_state_125    : run_state_t;
    signal delay            : std_logic_vector(1 downto 0);
    signal reset_state      : std_logic_vector(3 downto 0);

    signal fifo_wdata       : std_logic_vector(36*N_LINKS-1 downto 0);
    signal fifo_write       : std_logic_vector(N_LINKS-1 downto 0);

begin

    clk     <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);


    --! Setup
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    process(reset_n, clk)
    begin
    if ( reset_n = '0' ) then
        run_state_125   <= RUN_STATE_IDLE;
        reset_state     <= (others => '0');
		scifi_reg.addr  <= (others => '0');
        scifi_reg.wdata <= (others => '0');
        scifi_reg.re    <= '0';
        scifi_reg.we    <= '0';
        delay           <= (others => '0');
        --
    elsif rising_edge(clk) then
        
        delay <= delay + '1';
        scifi_reg.addr  <= (others => '0');
        scifi_reg.wdata <= (others => '0');
        scifi_reg.re    <= '0';
        scifi_reg.we    <= '0';
        
        if ( delay = "00" ) then

        case reset_state is
            
            -- first we enable the data generator config
            when "0000" =>
                scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DUMMY_REGISTER_W, 16));
                scifi_reg.wdata(11 downto 0)<= "000000000001";
                scifi_reg.we                <= '1';
                reset_state                 <= "0001";
                
            when "0001" =>
                scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DUMMY_REGISTER_W, 16));
                scifi_reg.wdata(11 downto 0)<= "000000000011";
                scifi_reg.we                <= '1';
                reset_state                 <= "0010";
            
            when "0010" =>
                scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DUMMY_REGISTER_W, 16));
                scifi_reg.wdata(11 downto 0)<= "000000001011";
                scifi_reg.we                <= '1';
                reset_state <= "0011";

            -- disable the PRBS decoder
            when "0011" =>
                scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DP_REGISTER_W, 16));
                scifi_reg.wdata(31)         <= '1';      -- i_SC_disable_dec PRBS
                scifi_reg.wdata(3 downto 0) <= "0100";   -- mask inputs
                scifi_reg.we                <= '1';
                reset_state <= "0100";
            
            when "0100" =>
                reset_state <= "0101";
            
            when "0101" =>
                reset_state <= "0110";
                
            when "0110" =>
                reset_state <= "0111";
            
            when "0111" =>
                reset_state <= "1000";
                
            when "1000" =>
                run_state_125 <= RUN_STATE_PREP;
                reset_state <= "1001";
                
            when "1001" =>
                run_state_125 <= RUN_STATE_SYNC;
                reset_state <= "1010";
                
            when "1010" =>
                run_state_125 <= RUN_STATE_RUNNING;
                --reset_state <= "1011";
                
            when "1011" =>
                reset_state <= "1100";
                
            when "1100" =>
                reset_state <= "0000";
                
            when others =>
                reset_state <= "0000";
                
        end case;
        end if;
    end if;
    end process;
    


    --! Scifi Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    -- scifi detector firmware
    e_scifi_path : entity work.scifi_path
    generic map (
        IS_SCITILE      => '0',
        N_MODULES       => N_MODULES,
        N_ASICS         => N_ASICS,
        N_LINKS         => N_LINKS,
        INPUT_SIGNFLIP  => x"FFFFFFFF", -- swap input 0 of con2 and 0 of con3 x"FFFFFFEE"
        LVDS_PLL_FREQ   => 125.0,
        LVDS_DATA_RATE  => 1250.0--,
    )
    port map (
        i_reg_addr                  => scifi_reg.addr(15 downto 0),
        i_reg_re                    => scifi_reg.re,
        o_reg_rdata                 => scifi_reg.rdata,
        i_reg_we                    => scifi_reg.we,
        i_reg_wdata                 => scifi_reg.wdata,

        o_chip_reset                => open,

        o_pll_test                  => open,
        i_data                      => (others => '0'),

        io_i2c_sda                  => open,
        io_i2c_scl                  => open,
        i_cec                       => '0',
        i_spi_miso                  => '0',
        i_i2c_int                   => '0',
        o_pll_reset                 => open,
        o_spi_scl                   => open,
        o_spi_mosi                  => open,

        o_fifo_write                => fifo_write,
        o_fifo_wdata                => fifo_wdata,

        i_common_fifos_almost_full  => (others => '0'),

        i_run_state                 => run_state_125,
        o_run_state_all_done        => open,

        o_MON_rxrdy                 => open,

        i_clk_ref_A                 => clk,
        i_clk_ref_B                 => clk,

        o_fast_pll_clk              => open,
        o_test_led                  => open,

        i_reset_156_n               => reset_n,
        i_clk_156                   => clk,
        i_reset_125_n               => reset_n,
        i_clk_125                   => clk--,
    );

    e_merger : entity work.data_merger
    generic map (
        N_LINKS     => 1,
        feb_mapping => (3,2,1,0)--,
    )
    port map (
        fpga_ID_in                 => x"000A",
        FEB_type_in                => "111000",

        run_state                  => run_state_125,
        run_number                 => (others => '0'),

        o_data_out                 => open,
        o_data_is_k                => open,

        slowcontrol_write_req      =>'0',
        i_data_in_slowcontrol      => (others => '0'),

        data_write_req(0)          => fifo_write(0),
        i_data_in                  => fifo_wdata(35 downto 0),
        o_fifos_almost_full        => open,

        override_data_in           => (others => '0'),
        override_data_is_k_in      => (others => '0'),
        override_req               => '0',
        override_granted           => open,

        can_terminate              => '0',
        o_terminated               => open,
        data_priority              => '0',
        o_rate_count               => open,

        reset                      => not reset_n,
        clk                        => clk--,
    );

end architecture;
