library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;


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

    signal pio_out_0                            : std_logic_vector(31 downto 0);
    signal pio_in_1                             : std_logic_vector(31 downto 0);
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
    signal spi_flash_cmdaddr_to_flash           : std_logic_vector(31 downto 0); 
    signal spi_flash_fifodata_to_flash          : std_logic_vector(31 downto 0);
    signal spi_flash_readfifo                   : std_logic;
    signal spi_flash_fifo_empty                 : std_logic;

    signal spi_flash_data_to_flash_nios         : std_logic_vector(7 downto 0);	
    signal writefromfifo                        : std_logic;
    signal fifowrite_last                       : std_logic;
    signal fiforead_last                        : std_logic;
    signal fifowriting                          : std_logic;
    signal fifo_read_pulse                      : std_logic;
    signal programmer_active                    : std_logic;

    -- SPI Arria Signals
    signal SPI_Nios_com                         : std_logic_vector(7 downto 0);
    signal SPI_Aria_data                        : std_logic_vector(31 downto 0);
    signal SPI_Max10_data                       : std_logic_vector(31 downto 0);
    signal SPI_addr_o                           : std_logic_vector(6 downto 0);
    signal SPI_rw                               : std_logic;

    -- SPI Arria ram
    signal ram_SPI_data                         : std_logic_vector(31 downto 0);
    signal SPI_ram_data                         : std_logic_vector(31 downto 0);
    signal SPI_ram_addr                         : std_logic_vector(13 downto 0);
    signal SPI_ram_rw                           : std_logic;

    -- ADC NIOS PIOs
    signal adc_data_0                           : std_logic_vector(31 downto 0);
    signal adc_data_1                           : std_logic_vector(31 downto 0);
    signal adc_data_2                           : std_logic_vector(31 downto 0);
    signal adc_data_3                           : std_logic_vector(31 downto 0);
    signal adc_data_4                           : std_logic_vector(31 downto 0);
   
   

begin

    -- signal defaults
    fpga_reset	<= '0'; 
    fpga_nconfig<= '1';
    reset_n     <= '1';
    mscb_ena    <= '0';
    attention_n <= "ZZ";
    
    -- SPI Arria10 to MAX10
    e_spi_secondary : entity work.spi_secondary
    generic map (
        SS      => '1', -- signal des ChipSelects 
        R       => '1', -- Read Signal
        lanes   => 4--,
        )
    port map(
        -- Max Data/register interface 
        o_Max_rw        => SPI_rw,
        o_Max_data      => SPI_Aria_data,
        o_Max_addr_o    => SPI_addr_o,
        o_b_addr        => SPI_Nios_com, -- command adrr.
        
        -- inputs
        i_Max_data      => SPI_Max10_data,
        i_SPI_cs        => fpga_spi_csn,
        i_SPI_clk       => fpga_spi_D3, -- fpga_spi_clk, --max10_osc_clk, --

        io_SPI_mosi     => fpga_spi_mosi,
        io_SPI_D1       => fpga_spi_D1,
        io_SPI_D2       => fpga_spi_D2,
        io_SPI_miso     => fpga_spi_miso,
        io_SPI_D3       => open--,
    );

    e_spi_decoder : entity work.spi_decoder
    port map(
        -- SPI secondary
        i_SPI_inst      => SPI_Nios_com,
        i_SPI_data      => SPI_Aria_data,
        i_SPI_addr_o    => SPI_addr_o,
        i_SPI_rw        => SPI_rw,	
        o_SPI_data      => SPI_Max10_data,

        -- ram interface
        i_ram_data      => ram_SPI_data,
        o_ram_data      => SPI_ram_data,
        o_ram_addr      => SPI_ram_addr,
        o_ram_rw        => SPI_ram_rw,

        -- ADC Nios PIOs
        i_adc_data_0    => adc_data_0,
        i_adc_data_1    => adc_data_1,
        i_adc_data_2    => adc_data_2,
        i_adc_data_3    => adc_data_3,
        i_adc_data_4    => adc_data_4,

        -- fifo interface
        i_fifo_data     => X"00000010",--fifo_SPI_data,
        o_fifo_data     => open,--SPI_fifo_data,
        o_fifo_next     => open,--SPI_fifo_next,
        o_fifo_rw       => open,--SPI_fifo_rw,
        
        -- command register --
        i_comm_data     => X"00000030",
        o_comm_data     => open,
        o_comm_rw       => open,
        
        -- status register--
        i_stat_data     => X"00000050",
        o_stat_data     => open,
        o_stat_rw       => open,

        -- register
        i_reg_data      => X"00000070",
        o_reg_data      => open,
        o_reg_addr      => open,
        o_reg_rw        => open--,
    );

    e_pll : entity work.pll
    port map(
        inclk0      => max10_osc_clk,
        c0          => clk10,
        c1          => clk100,
        c2          => clk50,
        locked      => pll_locked,--
    );

    e_nios : entity work.nios
        port map(
            clk_clk             => clk100,
            clk_0_clk           => clk100,-- clk_0.clk
            reset_0_reset_n     => '1' ,

            adc_data_0_export   => adc_data_0,-- adc_data_0.export
            adc_data_1_export   => adc_data_1,-- adc_data_1.export
            adc_data_2_export   => adc_data_2,-- adc_data_2.export
            adc_data_3_export   => adc_data_3,-- adc_data_3.export
            adc_data_4_export   => adc_data_4,-- adc_data_4.export

            merlin_master_translator_0_avalon_anti_master_0_address                  => SPI_ram_addr,                  --                   onchip_memory2_0_s2.address
			merlin_master_translator_0_avalon_anti_master_0_read                     => SPI_ram_rw,                   --                                      .write
			merlin_master_translator_0_avalon_anti_master_0_readdata                 => ram_SPI_data,                  --                                      .readdata
			merlin_master_translator_0_avalon_anti_master_0_write                    => '0',
			merlin_master_translator_0_avalon_anti_master_0_writedata                => SPI_ram_data,                  --                                      .writedata
			
			modular_adc_0_adc_pll_clock_clk           => clk10,
			modular_adc_0_adc_pll_locked_export       => pll_locked,
			
			pio_0_external_connection_export          => pio_out_0,
			pio_1_external_connection_export          => pio_in_1,
			
			reset_reset_n                             => reset_n,
			spi_flash_ctrl_external_connection_export => spi_flash_ctrl,
			
			spi_flash_datain_external_connection_export  => spi_flash_data_to_flash_nios,
			spi_flash_dataout_external_connection_export => spi_flash_data_from_flash,
			spi_flash_out_external_connection_export     => spi_flash_cmdaddr_to_flash,
			spi_flash_status_external_connection_export  => spi_flash_status,
			spi_flash_writefifo_clk_out_clk              => clk100,
			spi_flash_writefifo_out_readdata             => spi_flash_fifodata_to_flash,
			spi_flash_writefifo_out_read                 => spi_flash_readfifo,
			spi_flash_writefifo_out_waitrequest          => spi_flash_fifo_empty,               
			spi_flash_writefifo_reset_out_reset_n        => reset_n
		);

	
	
-- 	process(reset_n, clk100)
-- 	begin
-- 	if(reset_n = '0')then
-- 		writefromfifo <= '0';
-- 		fifowrite_last	<= '0';
-- 		fiforead_last  <= '0';
-- 		fifowriting		<= '0';
-- 		fifo_read_pulse <= '0';
-- 		programmer_active <= '0';
-- 	elsif(clk100'event and clk100 = '1')then
-- 		fifowrite_last <= spi_flash_ctrl(7);
-- 		fiforead_last	<= spi_flash_status(1);
-- 		fifo_read_pulse <= '0';
-- 		
-- 		if(spi_flash_ctrl(7) = '1' and fifowrite_last = '0' and programmer_active = '0') then
-- 			fifowriting <= '1';
-- 			fifo_read_pulse <= '1';	
-- 			wcounter <= (others => '0');
-- 		end if;
-- 		
-- 		if(fifowriting <= '1')then
-- 			if(spi_flash_fifo_empty = '1')then
-- 				fifowriting <= '0';
-- 			end if;
-- 		end if;
-- 		
-- 		if(spi_flash_readfifo = '1')then
-- 			wcounter <= wcounter + 1;
-- 		end if;
-- 		
-- 		if(programmer_active = '0' and fifowriting = '0' and 	
-- 			spi_busy = '0' and spi_flash_request_programmer = '1') then
-- 			spi_flash_granted_programmer <= '1';
-- 			programmer_active <= '1';
-- 		end if;
-- 		
-- 		if(programmer_active = '1' and spi_flash_request_programmer = '0') then
-- 			spi_flash_granted_programmer <= '0';
-- 			programmer_active <= '0';	
-- 	    end if;		
-- 	end if;
-- 	end process;
-- 	
-- 	pio_in_1(31 downto 16) <= std_logic_vector(wcounter);
-- 	
-- 	spi_strobe_nios 	<= spi_flash_ctrl(0);
-- 	spi_command_nios	<= spi_flash_cmdaddr_to_flash(31 downto 24);
-- 	spi_addr_nios		<= spi_flash_cmdaddr_to_flash(23 downto 0);
-- 	
-- 	spi_flash_data_to_flash <= spi_flash_data_to_flash_nios when fifowriting = '0'
-- 										else spi_flash_fifodata_to_flash(7 downto 0);
-- 										
-- 	spi_continue <= 	spi_continue_programmer when spi_flash_granted_programmer = '1'
-- 							else not spi_flash_fifo_empty	when fifowriting = '1'
-- 							else spi_flash_ctrl(1);
-- 							
-- 	spi_strobe 	<= spi_strobe_programmer when spi_flash_granted_programmer = '1'
-- 						else spi_strobe_nios;
-- 	spi_command 	<= spi_command_programmer when spi_flash_granted_programmer = '1'
-- 						else spi_command_nios;
-- 	spi_addr 		<= spi_addr_programmer when spi_flash_granted_programmer = '1'
-- 						else spi_addr_nios;							
-- 										
-- 	spi_flash_readfifo		<= spi_flash_status(1) or fifo_read_pulse;
-- 	
-- 	
-- 	
-- 	spi_flash_status(0) <= spi_ack;
-- 	spi_flash_status(1) <= spi_next_byte;
-- 	spi_flash_status(2) <= spi_byte_ready;
-- 	spi_flash_status(3) <= spi_busy;
-- 	spi_flash_status(6)		<= spi_flash_fifo_empty;
-- 	spi_flash_status(7)		<= fifowriting;
-- 
-- 	
-- 	
-- 	flash_if:spiflash
--     port map(
--         reset_n		=> reset_n,
--         clk				=> clk100,
-- 
--         spi_strobe 	=> spi_strobe,
--         spi_ack 		=> spi_ack,
-- 		  spi_busy		=> spi_busy,
--         spi_command 	=> spi_command,
--         spi_addr 		=> spi_addr,
--         spi_data 	   => spi_flash_data_to_flash,
--         spi_next_byte => spi_next_byte,
--         spi_continue  => spi_continue, 
--         spi_byte_out  => spi_flash_data_from_flash,
--         spi_byte_ready => spi_byte_ready,
-- 
--         spi_sclk		=> flash_sck,
--         spi_csn		=> flash_csn,
--         spi_mosi		=> flash_io0,
--         spi_miso		=> flash_io1,
--         spi_D2			=> flash_io2,
--         spi_D3     	=> flash_io3
--     );
-- 
-- 
-- 	 
-- 	 
-- 	 	programming_if:ps_programmer
--     port map(
--         reset_n     		=> reset_n,
--         clk         		=> clk100,
--         start           => pio_out_0(31),
--         start_address   => pio_out_0(23 downto 0),
-- 
--         Interface to SPI flash
--         spi_strobe      => spi_strobe_programmer,
--         spi_command     => spi_command_programmer,
--         spi_addr        => spi_addr_programmer,
--         spi_continue    => spi_continue_programmer,
--         spi_byte_out    => spi_flash_data_from_flash,
--         spi_byte_ready  => spi_byte_ready,
-- 
--         spi_flash_request   => spi_flash_request_programmer,
--         spi_flash_granted   => spi_flash_granted_programmer,
-- 
--         Interface to FPGA
--         fpga_conf_done		 => fpga_conf_done,
-- 		  fpga_nstatus		    => fpga_nstatus,
-- 		  fpga_nconfig			 => fpga_nconfig,
-- 		  fpga_data				 => fpga_data,
-- 		  fpga_clk				 => fpga_clk
--     );
-- 
-- 	 
-- 
end architecture rtl;
