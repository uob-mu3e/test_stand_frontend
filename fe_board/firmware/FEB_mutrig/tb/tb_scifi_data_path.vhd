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
    constant N_MODULES  : integer := 1;
    constant N_ASICS_TOTAL : natural := N_MODULES * N_ASICS;

    signal clk, reset_n : std_logic := '0';

    signal scifi_reg        : work.util.rw_t;    
    signal run_state_125    : run_state_t;
    signal delay            : std_logic_vector(1 downto 0);
    signal reset_state      : std_logic_vector(4 downto 0);
    
    signal i_simdata        : std_logic_vector(8*N_ASICS_TOTAL-1 downto 0) := (others => '0');
    signal i_simdatak       : std_logic_vector(N_ASICS_TOTAL-1 downto 0) := (others => '0');

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
        
        for i in 0 to N_ASICS_TOTAL-1 loop
            i_simdata((i+1)*8-1 downto i*8) <= x"BC";
            i_simdatak(i) <= '1';
        end loop;
        
        if ( delay = "00" ) then

        case reset_state is
            
            -- first we enable the data generator config
            when "00000" =>
                --scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DUMMY_REGISTER_W, 16));
                --scifi_reg.wdata(11 downto 0)<= "000000000001";
                --scifi_reg.we                <= '1';
                -- enable direct link readout
                scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_LINK_DATA_REGISTER_W, 16));
                scifi_reg.wdata(31)         <= '1';
                scifi_reg.wdata(3 downto 0) <= x"1";
                scifi_reg.we                <= '1';
                reset_state                 <= "00001";
                
            when "00001" =>
                --scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DUMMY_REGISTER_W, 16));
                --scifi_reg.wdata(11 downto 0)<= "000000000011";
                --scifi_reg.we                <= '1';
                reset_state                 <= "00010";
            
            when "00010" =>
                --scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DUMMY_REGISTER_W, 16));
                --scifi_reg.wdata(11 downto 0)<= "000000001011";
                --scifi_reg.we                <= '1';
                reset_state <= "00011";

            -- disable the PRBS decoder
            when "00011" =>
                scifi_reg.addr(15 downto 0) <= std_logic_vector(to_unsigned(SCIFI_CTRL_DP_REGISTER_W, 16));
                scifi_reg.wdata(31)         <= '1';      -- i_SC_disable_dec PRBS
                scifi_reg.wdata(3 downto 0) <= "0100";   -- mask inputs
                scifi_reg.we                <= '1';
                reset_state <= "00100";
            
            when "00100" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"1C";
                    i_simdatak(i) <= '1';
                end loop;
            
                reset_state <= "00101";
            
            when "00101" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"32";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "00110";
                
            when "00110" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"8F";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "00111";
            
            when "00111" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"50";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "01000";
                
            when "01000" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"01";
                    i_simdatak(i) <= '0';
                end loop;
                run_state_125 <= RUN_STATE_PREP;
                reset_state <= "01001";
                
            when "01001" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"CC";
                    i_simdatak(i) <= '0';
                end loop;
                run_state_125 <= RUN_STATE_SYNC;
                reset_state <= "01010";
                
            when "01010" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"5F";
                    i_simdatak(i) <= '0';
                end loop;
                run_state_125 <= RUN_STATE_RUNNING;
                reset_state <= "01011";
                
            when "01011" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"D0";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "01100";
                
            when "01100" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"C7";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "01101";
                
            when "01101" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"A5";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "01110";

            when "01110" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"69";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "01111";

            when "01111" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"52";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "10000";
                
            when "10000" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"68";
                    i_simdatak(i) <= '0';
                end loop;
                reset_state <= "10001";
                
            when "10001" =>
                for i in 0 to N_ASICS_TOTAL-1 loop
                    i_simdata((i+1)*8-1 downto i*8) <= x"9C";
                    i_simdatak(i) <= '1';
                end loop;
                reset_state <= "10010";
                
            when "10010" =>
                --
                
            when others =>
                reset_state <= "00000";
                
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
        
        -- simulation input
        i_enablesim                 => '1',
        i_simdata                   => i_simdata,
        i_simdatak                  => i_simdatak,

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
