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
        fpga_spi_D3             : inout std_logic;
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
    signal  control                             : std_logic_vector(31 downto 0);
    signal  spi_arria_we                        : std_logic;
    signal  rw_last                             : std_logic;
    signal  new_transaction                     : std_logic;

    signal flash_programming_ctrl               : std_logic_vector(31 downto 0);
    signal flash_w_cnt                          : std_logic_vector(31 downto 0);
    signal reset_n                              : std_logic;



    signal spi_flash_ctrl                       : std_logic_vector(7 downto 0); 
    signal spi_flash_status                     : std_logic_vector(7 downto 0); 
    signal spi_flash_data_from_flash            : std_logic_vector(7 downto 0);
    
    signal spi_flash_data_to_flash_nios         : std_logic_vector(7 downto 0);	
    signal spi_flash_cmdaddr_to_flash           : std_logic_vector(31 downto 0); 
	 signal spi_flash_fifo_data_nios					: std_logic_vector(8 downto 0);


    -- Fifo for programming data to the SPIflash
    signal spiflash_to_fifo_we                     : std_logic;
    signal spiflashfifo_empty                      : std_logic;
    signal spiflashfifo_full                       : std_logic;
    signal spiflashfifo_data_in                    : std_logic_vector(7 downto 0);
    signal spiflashfifo_data_out                   : std_logic_vector(7 downto 0);	 
    signal read_spiflashfifo                       : std_logic;
	 signal fifopiotoggle_last								: std_logic;

    -- spi arria
    signal SPI_inst                             : std_logic_vector(7 downto 0);
    signal SPI_Aria_data                        : std_logic_vector(31 downto 0);
    signal SPI_Max10_data                       : std_logic_vector(31 downto 0);
    signal SPI_addr_o                           : std_logic_vector(6 downto 0);
    signal SPI_rw                               : std_logic;

    signal spi_arria_addr                       : std_logic_vector(6 downto 0);
    signal spi_arria_addr_offset                : std_logic_vector(7 downto 0);
    signal spi_arria_rw                         : std_logic;
    signal spi_arria_data_to_arria              : std_logic_vector(31 downto 0);
    signal spi_arria_next_data                  : std_logic;
    signal spi_arria_word_from_arria            : std_logic_vector(31 downto 0);
    signal spi_arria_word_en                    : std_logic;
    signal spi_arria_byte_from_arria            : std_logic_vector(7 downto 0);
    signal spi_arria_byte_en                    : std_logic;

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
    
begin

    -- signal defaults, clk & resets
    -----------------------
    fpga_reset  <= '0';
    reset_n     <= pll_locked;
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
    status(1)  <= spi_arria_we;
    status(23 downto 2) <= (others => '0');
    status(31 downto 24) <= spi_flash_status;

    programming_status   <= (others => '0');

    -- SPI Arria10 to MAX10
    -----------------------
    e_spi_arria: entity work.spi_arria
        port map(
            ------ SPI
            i_SPI_csn       => fpga_spi_csn,
            i_SPI_clk       => fpga_spi_miso, -- replacement for missing connection 
            io_SPI_mosi     => fpga_spi_mosi,
            io_SPI_miso     => open,
            io_SPI_D1       => fpga_spi_D1,
            io_SPI_D2       => fpga_spi_D2,
            io_SPI_D3       => fpga_spi_D3, -- again, replacement
    
            clk100          => clk100,
            reset_n         => reset_n,
            addr            => spi_arria_addr,
            addroffset      => spi_arria_addr_offset,
            data_to_arria   => spi_arria_data_to_arria,
            rw              => spi_arria_rw,
            word_from_arria => spi_arria_word_from_arria,
            word_en         => spi_arria_word_en,
            byte_from_arria => spi_arria_byte_from_arria,
            byte_en         =>  spi_arria_byte_en
    );
 
    -- Write enable logic
    process(clk100, reset_n)
    begin
    if (reset_n = '0') then
        spi_arria_we <= '0';
        rw_last     <= '0';
        new_transaction <= '0';
    elsif(clk100'event and clk100 = '1')then
        rw_last     <= spi_arria_rw;
        if(spi_arria_addr = FEBSPI_ADDR_WRITENABLE
            and spi_arria_byte_from_arria = FEBSPI_PATTERN_WRITENABLE
            and spi_arria_byte_en = '1' and spi_arria_rw = '1') then
                spi_arria_we <= '1';
        end if;
        if(spi_arria_we = '1' and rw_last <= '0' and spi_arria_rw = '1') then
            new_transaction <= '1';
        end if;
        if(new_transaction = '1' and rw_last <= '1' and spi_arria_rw = '0') then
            spi_arria_we <= '0';
            new_transaction <= '0';
        end if;
    end if;
    end process;

    -- Multiplexer for data to_arria
    spi_arria_data_to_arria  
              <=   version when spi_arria_addr = FEBSPI_ADDR_GITHASH
                    else status when spi_arria_addr = FEBSPI_ADDR_STATUS
                    else control when  spi_arria_addr = FEBSPI_ADDR_CONTROL
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
                                     and spi_arria_addr_offset = X"04"
						  else (others => '0'); -- needed to avoid latch
                                     
                

    -- Write multiplexer
    process(clk100, reset_n)
    begin
    if (reset_n = '0') then
        control <= (others => '0');           
    elsif(clk100'event and clk100 = '1')then
        -- Word-wise writing
        if(spi_arria_rw = '1' and spi_arria_word_en = '1') then
            if(spi_arria_addr = FEBSPI_ADDR_CONTROL) then
                control <= spi_arria_word_from_arria;
            end if;
        end if;
        -- Byte-wise writing
        if(spi_arria_rw = '1' and spi_arria_byte_en = '1') then
            --if(spi_arria_addr = FEBSPI_ADDR_CONTROL) then
            --    control <= spi_arria_byte_from_arria;
            --end if;
        end if;
        
    end if;
    end process;

    -- NIOS
    -----------------------
    e_nios : component work.cmp.nios
    port map(
        -- clk & reset
        clk_clk                     => clk100,
        clk_spi_clk                 => clk100,
        rst_reset_n                 => reset_n,
        rst_spi_reset_n             => reset_n,

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
		  flash_fifo_data_export		=> spi_flash_fifo_data_nios
    );

 
e_flashprogramming_block: entity work.flashprogramming_block
    port map(
        clk100  	=> clk100,
        reset_n 	=> reset_n,
		  
		  control 	=> control,

        -- Flash SPI IF
        flash_csn               => flash_csn,
        flash_sck               => flash_sck,
        flash_io0               => flash_io0,
        flash_io1               => flash_io1,
        flash_io2               => flash_io2,
        flash_io3               => flash_io3,
        
        -- FPGA programming interface
        fpga_conf_done          => fpga_conf_done,
        fpga_nstatus            => fpga_nstatus,
        fpga_nconfig            => fpga_nconfig, 
        fpga_data               => fpga_data,
        fpga_clk                => fpga_clk,

        -- NIOS interface
        flash_programming_ctrl  => flash_programming_ctrl,
        flash_w_cnt             => flash_w_cnt,
        spi_flash_cmdaddr_to_flash  => spi_flash_cmdaddr_to_flash,
        spi_flash_ctrl          => spi_flash_ctrl,
        spi_flash_data_to_flash_nios => spi_flash_data_to_flash_nios,
        spi_flash_data_from_flash   => spi_flash_data_from_flash,
        spi_flash_status            => spi_flash_status,
		 spi_flash_fifo_data_nios     => spi_flash_fifo_data_nios, 
		 
		   -- Arria SPI interface
        spi_arria_byte_from_arria            => spi_arria_byte_from_arria,
        spi_arria_byte_en                    => spi_arria_byte_en,
        spi_arria_addr                       => spi_arria_addr
    );

 



end architecture arch;
