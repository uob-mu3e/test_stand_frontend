library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

use work.daq_constants.all;

entity top is 
    port (
        reset_max_bp_n          : in std_logic; -- Active low reset 
        max10_si_clk            : in std_logic; -- 50 MHZ clock from SI chip			//	SI5345
        max10_osc_clk           : in std_logic; -- 50 MHZ clock from oscillator		//	SI5345

        -- Flash SPI IF
        flash_csn               : out std_logic;
        flash_sck               : out std_logic;
        flash_io0               : inout std_logic;
        flash_io1               : inout std_logic;
        flash_io2               : inout std_logic;
        flash_io3               : inout std_logic;

        -- FPGA programming interface
        fpga_conf_done          : in std_logic;
        fpga_nstatus            : in std_logic;
        fpga_nconfig            : out std_logic;
        fpga_data               : out std_logic_vector(7 downto 0);
        fpga_clk                : out std_logic;
        fpga_reset              : out std_logic;

        -- SPI Interface to FPGA
        fpga_spi_clk            : in std_logic;
        fpga_spi_mosi           : inout std_logic;
        fpga_spi_miso           : inout std_logic;
        fpga_spi_D1             : inout std_logic;
        fpga_spi_D2             : inout std_logic;
        fpga_spi_D3             : in std_logic;
        fpga_spi_csn            : in std_logic;

        -- SPI Interface to backplane
        bp_spi_clk              : in std_logic;
        bp_spi_mosi             : in std_logic;
        bp_spi_miso             : out std_logic;
        bp_spi_miso_en          : out std_logic;
        bp_spi_csn              : in std_logic;

        -- Backplane signals
        board_select            : in std_logic;
        reset_cpu_backplane_n   : in std_logic;
        reset_fpga_bp_n         : in std_logic;
        bp_reset_fpga           : in std_logic;
        bp_mode_select          : in std_logic_vector(1 downto 0);
        mscb_out                : out std_logic;
        mscb_in                 : in std_logic;
        fpga_mscb_oe            : in std_logic;
        mscb_ena                : out std_logic;
        mscb_reset_n            : out std_logic; 
        ref_addr                : in std_logic_vector(7 downto 0);
        spi_adr                 : in std_logic_vector(2 downto 0);
        attention_n             : inout std_logic_vector(1 downto 0);
        temp_sens_dis           : out std_logic;
        spare                   : in std_logic_vector(2 downto 0)--;
);
end entity top;


architecture arch of top is

    signal clk100                               : std_logic;
    signal clk10                                : std_logic;
    signal clk50                                : std_logic;
    signal pll_locked                           : std_logic;

    signal  version                             : std_logic_vector(31 downto 0);
    signal  status                              : std_logic_vector(31 downto 0);
    signal  programming_status                  : std_logic_vector(31 downto 0);

    signal flash_programming_ctrl               : std_logic_vector(31 downto 0);
    signal flash_w_cnt                          : std_logic_vector(31 downto 0);
    signal reset_n                              : std_logic;

    -- SPI Flash
    signal spi_strobe_programmer                : std_logic;
    signal spi_command_programmer               : std_logic_vector(7 downto 0);
    signal spi_addr_programmer                  : std_logic_vector(23 downto 0);
    signal spi_continue_programmer              : std_logic;
    signal spi_flash_request_programmer         : std_logic;
    signal spi_flash_granted_programmer         : std_logic;

    signal spi_strobe_nios                      : std_logic;
    signal spi_command_nios                     : std_logic_vector(7 downto 0);
    signal spi_addr_nios                        : std_logic_vector(23 downto 0);
    signal spi_continue_nios                    : std_logic;

    signal spi_ack                              : std_logic;
    signal spi_busy                             : std_logic;
    signal spi_next_byte                        : std_logic;
    signal spi_byte_ready                       : std_logic;

    signal spi_strobe                           : std_logic;
    signal spi_continue                         : std_logic;
    signal spi_command                          : std_logic_vector(7 downto 0);
    signal spi_addr                             : std_logic_vector(23 downto 0);

    signal spi_flash_ctrl                       : std_logic_vector(7 downto 0); 
    signal spi_flash_status                     : std_logic_vector(7 downto 0); 
    signal spi_flash_data_from_flash            : std_logic_vector(7 downto 0);
    signal spi_flash_data_to_flash              : std_logic_vector(7 downto 0);
    signal spi_flash_data_to_flash_nios         : std_logic_vector(7 downto 0);	
    signal spi_flash_cmdaddr_to_flash           : std_logic_vector(31 downto 0); 
    signal spi_flash_fifodata_to_flash          : std_logic_vector(31 downto 0);
    signal spi_flash_readfifo                   : std_logic;
    signal spi_flash_fifo_empty                 : std_logic;

    type spiflashstate_type is (idle, fifowriting, programming);
    signal spiflashstate : spiflashstate_type;
    signal fifo_req_last                        : std_logic;
    signal fifo_read_pulse                      : std_logic;
    signal wcounter                             : std_logic_vector(15 downto 0);

    -- Fifo for programming data from Arria
    signal arria_to_fifo_we                     : std_logic;
    signal arriafifo_empty                      : std_logic;
    signal arriafifo_full                       : std_logic;
    signal arriafifo_data                       : std_logic_vector(7 downto 0);
    signal read_arriafifo                       : std_logic;

    -- spi arria
    signal SPI_inst                         : std_logic_vector(7 downto 0);
    signal SPI_Aria_data                        : std_logic_vector(31 downto 0);
    signal SPI_Max10_data                       : std_logic_vector(31 downto 0);
    signal SPI_addr_o                           : std_logic_vector(6 downto 0);
    signal SPI_rw                               : std_logic;

    signal spi_arria_addr           : std_logic_vector(6 downto 0);
    signal spi_arria_addr_offset    : std_logic_vector(7 downto 0);
    signal spi_arria_rw             : std_logic;
    signal spi_arria_data_to_arria  : std_logic_vector(31 downto 0);
    signal spi_arria_next_data      : std_logic;
    signal spi_arria_word_from_arria: std_logic_vector(31 downto 0);
    signal spi_arria_word_en        : std_logic;
    signal spi_arria_byte_from_arria: std_logic_vector(7 downto 0);
    signal spi_arria_byte_en        : std_logic;

    -- spi arria ram
    signal ram_SPI_data                         : std_logic_vector(31 downto 0);
    signal SPI_ram_data                         : std_logic_vector(31 downto 0);
    signal SPI_ram_addr                         : std_logic_vector(13 downto 0);
    signal SPI_ram_rw                           : std_logic;

    -- adc nios
    signal adc_data_0                           : std_logic_vector(31 downto 0);
    signal adc_data_1                           : std_logic_vector(31 downto 0);
    signal adc_data_2                           : std_logic_vector(31 downto 0);
    signal adc_data_3                           : std_logic_vector(31 downto 0);
    signal adc_data_4                           : std_logic_vector(31 downto 0);
    
    COMPONENT sssscfifo
    GENERIC (
            add_ram_output_register         : STRING;
            intended_device_family          : STRING;
            lpm_numwords            : NATURAL;
            lpm_showahead           : STRING;
            lpm_type                : STRING;
            lpm_width               : NATURAL;
            lpm_widthu              : NATURAL;
            overflow_checking               : STRING;
            underflow_checking              : STRING;
            use_eab         : STRING
    );
    PORT (
                    aclr    : IN STD_LOGIC ;
                    clock   : IN STD_LOGIC ;
                    data    : IN STD_LOGIC_VECTOR (253 DOWNTO 0);
                    rdreq   : IN STD_LOGIC ;
                    sclr    : IN STD_LOGIC ;
                    wrreq   : IN STD_LOGIC ;
                    almost_full     : OUT STD_LOGIC ;
                    empty   : OUT STD_LOGIC ;
                    full    : OUT STD_LOGIC;
                    q       : OUT STD_LOGIC_VECTOR (253 DOWNTO 0)
    );
    END COMPONENT;

begin

    -- signal defaults, clk & resets
    -----------------------
    fpga_reset  <= '0';
    reset_n     <= '1';
    mscb_ena    <= '0';
    attention_n <= "ZZ";

    e_pll : entity work.ip_altpll
    port map(
        inclk0      => max10_osc_clk,
        c0          => clk10,
        c1          => clk100,
        c2          => clk50,
        locked      => pll_locked--,
    );

    e_vreg: entity work.version_reg
    port map(
        data_out => version(27 downto 0)
    );
    version(31 downto 28) <= (others => '0');

    status(0)  <= pll_locked;
    status(23 downto 1) <= (others => '0');
    status(31 downto 24) <= spi_flash_status;

    programming_status   <= (others => '0');

    -- SPI Arria10 to MAX10
    -----------------------
    e_spi_arria: entity work.spi_arria
        port map(
            ------ SPI
            i_SPI_cs        => fpga_spi_csn,
            i_SPI_clk       => fpga_spi_D3, -- replacement for missing connection 
            io_SPI_mosi     => fpga_spi_mosi,
            io_SPI_miso     => open,
            io_SPI_D1       => fpga_spi_D1,
            io_SPI_D2       => fpga_spi_D2,
            io_SPI_D3       => fpga_spi_miso, -- again, replacement
    
            clk100          => clk100,
            reset_n         => reset_n,
            addr            => spi_arria_addr,
            addroffset      => spi_arria_addr_offset,
            data_to_arria   => spi_arria_data_to_arria,
            rw              => spi_arria_rw,
            word_en         => spi_arria_word_en,
            byte_from_arria => spi_arria_byte_from_arria,
            byte_en         =>  spi_arria_byte_en
    );
 
    -- Multiplexer for data to_arria
    spi_arria_data_to_arria  
              <=   version when spi_arria_addr = FEBSPI_ADDR_GITHASH
                    else status when spi_arria_addr = FEBSPI_ADDR_STATUS
                    else programming_status when spi_arria_addr = FEBSPI_ADDR_PROGRAMMING_STATUS
                    else flash_w_cnt when spi_arria_addr = FEBSPI_ADDR_PROGRAMMING_COUNT
                    else adc_data_0 when spi_arria_addr = FEBSPI_ADDR_ADCDATA
                                     and spi_arria_addr_offset = X"00"
                    else adc_data_1 when spi_arria_addr = FEBSPI_ADDR_ADCDATA
                                     and spi_arria_addr_offset = X"01"
                    else adc_data_2 when spi_arria_addr = FEBSPI_ADDR_ADCDATA
                                     and spi_arria_addr_offset = X"02"
                    else adc_data_3 when spi_arria_addr = FEBSPI_ADDR_ADCDATA
                                     and spi_arria_addr_offset = X"03"
                    else adc_data_4 when spi_arria_addr = FEBSPI_ADDR_ADCDATA
                                     and spi_arria_addr_offset = X"04";
                                     
    arria_to_fifo_we <= spi_arria_byte_en when spi_arria_addr = FEBSPI_ADDR_PROGRAMMING_WFIFO
                else '0';                   

    -- NIOS
    -----------------------
    e_nios : component work.cmp.nios
    port map(
        -- clk & reset
        clk_clk                     => clk100,
        clk_spi_clk                 => clk100,
        clk_flash_fifo_clk          => clk100,
        rst_reset_n                 => reset_n,
        rst_spi_reset_n             => reset_n,
        reset_flash_fifo_reset_n    => reset_n,

        -- generic pio
        pio_export                  => open,

        -- adc
        adc_pll_clock_clk           => clk10,
        adc_pll_locked_export       => pll_locked,
        adc_d0_export               => adc_data_0,
        adc_d1_export               => adc_data_1,
        adc_d2_export               => adc_data_2,
        adc_d3_export               => adc_data_3,
        adc_d4_export               => adc_data_4,

        -- arria spi
        ava_mm_address              => SPI_ram_addr,
        ava_mm_read                 => SPI_ram_rw,
        ava_mm_readdata             => ram_SPI_data,
        ava_mm_write                => '0',
        ava_mm_writedata            => SPI_ram_data,

        -- spi not used at the moment
        spi_MISO                    => '1',
        spi_MOSI                    => open,
        spi_SCLK                    => open,
        spi_SS_n                    => open,

        -- i2c not used at the moment
        i2c_sda_in                  => '1',
        i2c_scl_in                  => '1',
        i2c_sda_oe                  => open,
        i2c_scl_oe                  => open,

        -- flash spi
        flash_ps_ctrl_export        => flash_programming_ctrl,
        flash_w_cnt_export          => flash_w_cnt,
        flash_cmd_addr_export       => spi_flash_cmdaddr_to_flash,
        flash_ctrl_export           => spi_flash_ctrl,
        flash_i_data_export         => spi_flash_data_to_flash_nios,
        flash_o_data_export         => spi_flash_data_from_flash,
        flash_status_export         => spi_flash_status,
        out_flash_fifo_readdata     => spi_flash_fifodata_to_flash,
        out_flash_fifo_read         => spi_flash_readfifo,
        out_flash_fifo_waitrequest  => spi_flash_fifo_empty--,
    );

 

    process(reset_n, clk100)
    begin
    if ( reset_n = '0' ) then
        spiflashstate                   <= idle;
        spi_flash_granted_programmer    <= '0';
        fifo_read_pulse                 <= '0';
        fifo_req_last                   <= '0';
    elsif ( clk100'event and clk100 = '1' ) then

        fifo_read_pulse                 <= '0';
        fifo_req_last                   <= spi_flash_ctrl(7);

        case spiflashstate is
        when idle =>
            if (spi_busy = '0' and spi_flash_request_programmer = '1' ) then
                spiflashstate <= programming;
            end if;

            if ( spi_flash_ctrl(7) = '1' and  fifo_req_last = '0') then
                spiflashstate   <= fifowriting;
                fifo_read_pulse <= '1';
                wcounter        <= (others => '0');
            end if;
        when fifowriting =>
            wcounter                        <= wcounter + 1;
            if ( spi_flash_fifo_empty = '1' ) then
                spiflashstate <= idle;
            end if;    
        when programming =>
            spi_flash_granted_programmer    <= '1';
            if(spi_flash_request_programmer = '0') then
                spiflashstate <= idle;
            end if;
        when others =>
            spiflashstate <= idle;
        end case;
    end if;
    end process;

    flash_w_cnt(31 downto 16)   <= std_logic_vector(wcounter);
    
    spi_strobe_nios             <= spi_flash_ctrl(0);
    spi_command_nios            <= spi_flash_cmdaddr_to_flash(31 downto 24);
    spi_addr_nios               <= spi_flash_cmdaddr_to_flash(23 downto 0);
    
    spi_flash_readfifo          <= spi_next_byte or fifo_read_pulse;

    spi_flash_data_to_flash     <= spi_flash_fifodata_to_flash(7 downto 0) when spiflashstate = fifowriting
                                  else  spi_flash_data_to_flash_nios;

    spi_continue                <= spi_continue_programmer  when spiflashstate = programming
                                else not spi_flash_fifo_empty when spiflashstate = fifowriting
                                else spi_flash_ctrl(1);

    spi_strobe                  <= spi_strobe_programmer when spiflashstate = programming
                                   else spi_strobe_nios;
    spi_command                 <= spi_command_programmer when spiflashstate = programming
                                   else spi_command_nios;
    spi_addr                    <= spi_addr_programmer when spiflashstate = programming
                                   else spi_addr_nios;

    spi_flash_status(0)         <= spi_ack;
    spi_flash_status(1)         <= spi_next_byte;
    spi_flash_status(2)         <= spi_byte_ready;
    spi_flash_status(3)         <= spi_busy;
    spi_flash_status(6)         <= spi_flash_fifo_empty;
    spi_flash_status(7)         <= '1' when spiflashstate = fifowriting
                                    else '0';


    e_spiflash : entity work.spiflash
    port map(
        -- clk & reset
        reset_n         => reset_n,
        clk             => clk100,
        -- spi ctrl
        spi_strobe      => spi_strobe,
        spi_ack         => spi_ack,
        spi_busy        => spi_busy,
        spi_command     => spi_command,
        spi_addr        => spi_addr,
        spi_data        => spi_flash_data_to_flash,
        spi_next_byte   => spi_next_byte,
        spi_continue    => spi_continue, 
        spi_byte_out    => spi_flash_data_from_flash,
        spi_byte_ready  => spi_byte_ready,
        -- spi to flash
        spi_sclk        => flash_sck,
        spi_csn         => flash_csn,
        spi_mosi        => flash_io0,
        spi_miso        => flash_io1,
        spi_D2          => flash_io2,
        spi_D3          => flash_io3--,
    );

    programming_if : entity work.fpp_programmer
    port map(
        -- clk & reset
        reset_n             => reset_n,
        clk                 => clk100,
        -- spi addr
        start               => flash_programming_ctrl(31),
        start_address       => flash_programming_ctrl(23 downto 0),
        --Interface to SPI flash
        spi_strobe          => spi_strobe_programmer,
        spi_command         => spi_command_programmer,
        spi_addr            => spi_addr_programmer,
        spi_continue        => spi_continue_programmer,
        spi_byte_out        => spi_flash_data_from_flash,
        spi_byte_ready      => spi_byte_ready,
        spi_flash_request   => spi_flash_request_programmer,
        spi_flash_granted   => spi_flash_granted_programmer,
        --Interface to FPGA
        fpga_conf_done      => fpga_conf_done,
        fpga_nstatus        => fpga_nstatus,
        fpga_nconfig        => fpga_nconfig,
        fpga_data           => fpga_data,
        fpga_clk            => fpga_clk--,
    );

    scfifo_component : altera_mf.altera_mf_components.scfifo
        GENERIC MAP (
                add_ram_output_register => "ON",
                intended_device_family => "Max 10",
                lpm_numwords => 256,
                lpm_showahead => "OFF",
                lpm_type => "scfifo",
                lpm_width => 8,
                lpm_widthu => 8,
                overflow_checking => "ON",
                underflow_checking => "ON",
                use_eab => "ON"
        )
        PORT MAP (
                aclr => '0',
                clock => clk100,
                data => spi_arria_byte_from_arria,
                rdreq => read_arriafifo,
                sclr => not reset_n,
                wrreq => arria_to_fifo_we,
                empty => arriafifo_empty,
                full  => arriafifo_full,
                q => arriafifo_data
        );

end architecture arch;
