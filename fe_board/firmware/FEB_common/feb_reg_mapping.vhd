-- M.Mueller, November 2020

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;
use work.feb_sc_registers.all;

entity feb_reg_mapping is
generic (
    N_LINKS : positive := 1--;
);
port (
    i_clk_156                   : in  std_logic;
    i_reset_n                   : in  std_logic;

    i_reg_add                   : in  std_logic_vector(7 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    -- inputs  156--------------------------------------------
    -- ALL INPUTS DEFAULT TO (n*4-1 downto 0 => x"CCC..", others => '1')
    i_run_state_156             : in  run_state_t                   := RUN_STATE_UNDEFINED;
    i_merger_rate_count         : in  std_logic_vector(31 downto 0) := x"CCCCCCCC";
    i_reset_phase               : in  std_logic_vector(15 downto 0) := x"CCCC";
    i_arriaV_temperature        : in  std_logic_vector( 7 downto 0) := x"CC";
    i_fpga_type                 : in  std_logic_vector( 5 downto 0) := "111100";
    i_adc_reg                   : in  reg32array_t    ( 4 downto 0) := (others => x"CCCCCCCC");

    -- outputs 156--------------------------------------------
    o_reg_cmdlen                : out std_logic_vector(31 downto 0);
    o_reg_offset                : out std_logic_vector(31 downto 0);
    o_reg_reset_bypass          : out std_logic_vector(15 downto 0);
    o_reg_reset_bypass_payload  : out std_logic_vector(31 downto 0);
    o_arriaV_temperature_clr    : out std_logic;
    o_arriaV_temperature_ce     : out std_logic;
    o_fpga_id_reg               : out std_logic_vector(N_LINKS*16-1 downto 0)--;

);
end entity;

architecture rtl of feb_reg_mapping is
-- outputs
    signal reg_cmdlen               : std_logic_vector(31 downto 0);
    signal reg_offset               : std_logic_vector(31 downto 0);
    signal reg_reset_bypass         : std_logic_vector(15 downto 0);
    signal reg_reset_bypass_payload : std_logic_vector(31 downto 0);
    signal fpga_id_reg              : std_logic_vector(N_LINKS*16-1 downto 0);

-- inputs
    signal run_state_156            : run_state_t;
    signal merger_rate_count        : std_logic_vector(31 downto 0);
    signal reset_phase              : std_logic_vector(15 downto 0);
    signal arriaV_temperature       : std_logic_vector( 7 downto 0);
    signal fpga_type                : std_logic_vector( 5 downto 0);
    signal adc_reg                  : reg32array_t    ( 4 downto 0);

-- R/W test signals
    signal test_read                : std_logic;
    signal test_write               : std_logic;
    signal test_read_data           : std_logic_vector(31 downto 0);
    signal test_write_data          : std_logic_vector(31 downto 0);

begin

    o_reg_cmdlen                <= reg_cmdlen;
    o_reg_offset                <= reg_offset;
    o_reg_reset_bypass          <= reg_reset_bypass;
    o_reg_reset_bypass_payload  <= reg_reset_bypass_payload;
    o_fpga_id_reg               <= fpga_id_reg;

    process(i_clk_156)

    variable regaddr : integer;

    begin
    if rising_edge(i_clk_156) then
   
        o_reg_rdata         <= X"CCCCCCCC";
        regaddr             := to_integer(unsigned(i_reg_add(7 downto 0)));
        test_read           <= '0';
        test_write          <= '0';

        -- cmdlen
        if ( regaddr = CMD_LEN_REGISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata <= reg_cmdlen;
        end if;
        if ( regaddr = CMD_LEN_REGISTER_RW and i_reg_we = '1' ) then
            reg_cmdlen <= i_reg_wdata;
        end if;

        -- offset
        if ( regaddr = CMD_OFFSET_REGISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata <= reg_offset;
        end if;
        if ( regaddr = CMD_OFFSET_REGISTER_RW and i_reg_we = '1' ) then
            reg_offset <= i_reg_wdata;
        end if;

        -- reset bypass
        if ( regaddr = RUN_STATE_RESET_BYPASS_REGISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata(15 downto 0) <= reg_reset_bypass(15 downto 0);
            o_reg_rdata(16+9 downto 16) <= i_run_state_156;
        end if;
        if ( regaddr = RUN_STATE_RESET_BYPASS_REGISTER_RW and i_reg_we = '1' ) then
            reg_reset_bypass(15 downto 0) <= i_reg_wdata(15 downto 0); -- upper bits are read-only status
        end if;

        -- reset payload
        if ( regaddr = RESET_PAYLOAD_RGEISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata <= reg_reset_bypass_payload;
        end if;
        if ( regaddr = RESET_PAYLOAD_RGEISTER_RW and i_reg_we = '1' ) then
            reg_reset_bypass_payload <= i_reg_wdata;
        end if;

        -- rate measurement
        if ( regaddr = MERGER_RATE_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= i_merger_rate_count;
        end if;

        -- reset phase
        if ( regaddr = RESET_PHASE_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= x"0000" & i_reset_phase;
        end if;

        -- ArriaV temperature
        if ( regaddr = ARRIA_TEMP_REGISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata <= x"000000" & i_arriaV_temperature;
        end if;
        if ( regaddr = ARRIA_TEMP_REGISTER_RW and i_reg_we = '1' ) then
            o_arriaV_temperature_clr  <= i_reg_wdata(0);
            o_arriaV_temperature_ce   <= i_reg_wdata(1);
        end if;

        -- mscb

        -- git head hash
        if ( regaddr = GIT_HASH_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata <= work.cmp.GIT_HEAD(0 to 31);
        end if;
        -- fpga id
        if ( regaddr = FPGA_ID_REGISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(fpga_id_reg'range) <= fpga_id_reg;
        end if;
        if ( regaddr = FPGA_ID_REGISTER_RW and i_reg_we = '1' ) then
            o_reg_rdata <= (others => '0');
            fpga_id_reg(N_LINKS*16-1 downto 0) <= i_reg_wdata(N_LINKS*16-1 downto 0);
        end if;
        -- fpga type
        if ( regaddr = FPGA_TYPE_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(i_fpga_type'range) <= i_fpga_type;
        end if;
        --max ADC data--
        if ( regaddr = MAX10_ADC_0_1_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= i_adc_reg(0);
        end if;
        if ( regaddr = MAX10_ADC_2_3_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= i_adc_reg(1);
        end if;
        if ( regaddr = MAX10_ADC_4_5_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= i_adc_reg(2);
        end if;
        if ( regaddr = MAX10_ADC_6_7_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= i_adc_reg(3);
        end if;
        if ( regaddr = MAX10_ADC_8_9_REGISTER_R and i_reg_re = '1' ) then
            o_reg_rdata <= i_adc_reg(4);
        end if;
        --TODO: Fireflies


        -- NON-incrementing reads/writes TEST
        if ( regaddr = NONINCREMENTING_TEST_REGISTER_RW and i_reg_re = '1' ) then
            o_reg_rdata <= test_read_data;
            test_read   <= '1';
        end if;
        if ( regaddr = NONINCREMENTING_TEST_REGISTER_RW and i_reg_we = '1' ) then
            test_write_data <= i_reg_wdata;
            test_write      <= '1';
        end if;

    end if;
    end process;
    
    nonincrementing_rw_test_fifo: entity work.ip_scfifo
    generic map(
        ADDR_WIDTH      => 4,
        DATA_WIDTH      => 32,
        SHOWAHEAD       => "ON",
        DEVICE          => "Arria V"--,
    )
    port map (
        clock           => i_clk_156,
        sclr            => '0',
        data            => test_write_data,
        wrreq           => test_write,
        q               => test_read_data,
        rdreq           => test_read--,
    );
    
end architecture;
