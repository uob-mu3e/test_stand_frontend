library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

use work.daq_constants.all;
use work.spiflash_commands.all;

entity flashprogramming_block is
    port(
        clk100  : in std_logic;
        reset_n : in std_logic;

        control                 : in std_logic_vector(31 downto 0);
        status                  : out std_logic_vector(31 downto 0);

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

        fpp_crclocation         : out std_logic_vector(31 downto 0);

        -- NIOS interface
        flash_programming_ctrl  : in std_logic_vector(31 downto 0);
        flash_w_cnt             : out std_logic_vector(31 downto 0);
        spi_flash_cmdaddr_to_flash  : in std_logic_vector(31 downto 0);
        spi_flash_ctrl          : in std_logic_vector(7 downto 0);
        spi_flash_data_to_flash_nios : in std_logic_vector(7 downto 0);
        spi_flash_data_from_flash   : out std_logic_vector(7 downto 0);
        spi_flash_status            : out std_logic_vector(7 downto 0);
        spi_flash_fifo_data_nios    : in std_logic_vector(8 downto 0);
        
        -- Arria SPI interface
        spi_arria_byte_from_arria            : in std_logic_vector(7 downto 0);
        spi_arria_byte_en                    : in std_logic;     
        spi_arria_addr                       : in std_logic_vector(6 downto 0);
        addr_from_arria                      : in std_logic_vector(23 downto 0)
    );
end entity flashprogramming_block;

architecture RTL of flashprogramming_block is
        -- SPI Flash
        signal spi_strobe_programmer                : std_logic;
        signal spi_command_programmer               : std_logic_vector(7 downto 0);
        signal spi_addr_programmer                  : std_logic_vector(23 downto 0);
        signal spi_continue_programmer              : std_logic;
        signal spi_flash_request_programmer         : std_logic;
        signal spi_flash_granted_programmer         : std_logic;
    
        signal spi_flash_data_to_flash              : std_logic_vector(7 downto 0);
        signal spi_flash_data_from_flash_int        : std_logic_vector(7 downto 0);

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

        type spiflashstate_type is (idle, fifowriting, 
            arriawriting1, arriawriting2, arriawriting3, arriawriting4, arriawriting5,
            arriawriting6, arriawriting7, arriafifowriting, arriawriting8, programming);
        signal spiflashstate : spiflashstate_type;
        signal fifo_req_last                        : std_logic;
        signal arria_write_req_last                  : std_logic;
        signal fifo_read_pulse                      : std_logic;
        signal wcounter                             : std_logic_vector(15 downto 0);

        -- Fifo for programming data to the SPIflash
        signal spiflash_to_fifo_we                     : std_logic;
        signal spiflashfifo_empty                      : std_logic;
        signal spiflashfifo_full                       : std_logic;
        signal spiflashfifo_data_in                    : std_logic_vector(7 downto 0);
        signal spiflashfifo_data_out                   : std_logic_vector(7 downto 0);	 
        signal read_spiflashfifo                       : std_logic;
	    signal fifopiotoggle_last					   : std_logic;

        signal arriawriting                            : std_logic;
        signal wipseen                                 : std_logic;

        signal spi_strobe_arria                      : std_logic;
        signal spi_command_arria                     : std_logic_vector(7 downto 0);
        signal spi_addr_arria                        : std_logic_vector(23 downto 0);
        signal spi_continue_arria                    : std_logic;

        signal fpp_crcerror                         : std_logic;
        signal fpp_timeout                          : std_logic;
        signal fpp_debug                            : std_logic_vector(7 downto 0);

        signal addrlast                             : std_logic_vector(23 downto 0);

begin

    spi_flash_data_from_flash <= spi_flash_data_from_flash_int;

process(reset_n, clk100)
begin
if ( reset_n = '0' ) then
    spiflashstate                   <= idle;
    spi_flash_granted_programmer    <= '0';
    fifo_req_last                   <= '0';
    arria_write_req_last             <= '0';
    spiflash_to_fifo_we				<= '0';
    fifopiotoggle_last			    <= '0';
    arriawriting                    <= '0';
    spi_strobe_arria                <= '0';
    spi_continue_arria              <= '0';
    status                          <= (others => '0');
    addrlast                        <= (others => '0');


elsif ( clk100'event and clk100 = '1' ) then
    fifopiotoggle_last			    <= spi_flash_fifo_data_nios(8);
    fifo_req_last                   <= spi_flash_ctrl(7);
    arria_write_req_last             <= control(0);
    spiflash_to_fifo_we 	     	<= '0';
      

    status(0) <= arriawriting;
    status(1) <= spi_busy;

    status(16)              <= fpga_conf_done;
    status(17)              <= fpga_nstatus;
    status(18)              <= fpp_timeout;
    status(19)              <= fpp_crcerror;
    status(23 downto 20)    <= (others => '0');    
    status(31 downto 24)    <= fpp_debug;

    status(15 downto 2) <= (others => '0');

    case spiflashstate is
    when idle =>
        if (spi_busy = '0' and spi_flash_request_programmer = '1' ) then
            spiflashstate <= programming;
        end if;

        if ( spi_flash_ctrl(7) = '1' and  fifo_req_last = '0') then
            spiflashstate   <= fifowriting;
            wcounter        <= (others => '0');
        end if;

        if(control(0) = '1' and arria_write_req_last = '0') then -- here we start the sequence for erasing 
                                                                 -- and writing an spi flash block
                                                                 -- we only erase if we just passed a 64K block boundary

            addrlast <= addr_from_arria;                                                
            if(addr_from_arria(15 downto 0) = X"0000") then                                                    
                spiflashstate  <= arriawriting1;
                arriawriting   <= '1';
            else 
                spiflashstate  <= arriawriting5;
                arriawriting   <= '1';
            end if;    

            --if((unsigned(addr_from_arria) /= 0 and unsigned(addr_from_arria) - unsigned(addrlast) /= 256) or spiflashfifo_empty = '1')then
            --    status(31 downto 20) <= addr_from_arria(23 downto 12);
            --end if;    

        end if;    

            
        if(spi_arria_byte_en = '1' and spi_arria_addr = FEBSPI_ADDR_PROGRAMMING_WFIFO) then
            spiflash_to_fifo_we <= '1';
            spiflashfifo_data_in <= spi_arria_byte_from_arria;
        elsif(fifopiotoggle_last /= spi_flash_fifo_data_nios(8)) then
            spiflash_to_fifo_we <= '1';
            spiflashfifo_data_in <= spi_flash_fifo_data_nios(7 downto 0);
        end if;

    when arriawriting1 =>    -- set the write enable
        status(2)               <= '1';
        spi_command_arria       <= COMMAND_WRITE_ENABLE;
        spi_addr_arria          <= (others => '0');
        spi_continue_arria      <= '0';
        spi_strobe_arria        <= '1';
        if(spi_ack = '1')then
            spiflashstate <= arriawriting2;
            spi_strobe_arria        <= '0';
        end if;
    when arriawriting2 =>  -- send the erase command
        status(3)               <= '1';
        spi_command_arria       <= COMMAND_BLOCK_ERASE_64;
        spi_addr_arria          <= addr_from_arria;
        spi_continue_arria      <= '0';
        spi_strobe_arria        <= '1';
        if(spi_ack = '1')then
            spiflashstate <= arriawriting3;
            spi_strobe_arria        <= '0';
            wipseen                 <= '0';
        end if;
    when arriawriting3 => -- wait for the WIP bit to go off 
        status(4)               <= '1';
        spi_command_arria       <= COMMAND_READ_STATUS_REGISTER1;
        spi_addr_arria          <= (others => '0');
        spi_continue_arria      <= '1';
        spi_strobe_arria        <= '1';
        if(spi_byte_ready = '1' and spi_flash_data_from_flash_int(0) = '1') then
            wipseen <= '1';
        end if;

        if(spi_byte_ready = '1' and spi_flash_data_from_flash_int(0) = '0' and wipseen = '1') then
            spiflashstate           <= arriawriting4;
            spi_continue_arria      <= '0';
            spi_strobe_arria        <= '0'; 
        end if;
    when arriawriting4 => -- make sure spi is ready again
        status(5)               <= '1';
        if(spi_busy <= '0')then
            spiflashstate           <= arriawriting5;
        end if;
    when arriawriting5 =>  -- set the write enable
        status(6)               <= '1';
        spi_command_arria       <= COMMAND_WRITE_ENABLE;
        spi_addr_arria          <= (others => '0');
        spi_continue_arria      <= '0';
        spi_strobe_arria        <= '1';
        if(spi_ack = '1')then
            spiflashstate <= arriawriting6;
            spi_strobe_arria        <= '0';
        end if;
    when arriawriting6 => -- check if we set the write enable successfully 
        status(7)               <= '1';
        spi_command_arria       <= COMMAND_READ_STATUS_REGISTER1;
        spi_addr_arria          <= (others => '0');
        spi_continue_arria      <= '0';
        spi_strobe_arria        <= '1';
        if(spi_byte_ready = '1' and spi_flash_data_from_flash_int(1) = '1') then
            spiflashstate <= arriawriting7;
            spi_strobe_arria        <= '0';
        elsif(spi_byte_ready = '1' and spi_flash_data_from_flash_int(1) = '0') then
            spiflashstate <= arriawriting5;  -- try setting write enable again
            spi_strobe_arria        <= '0';
        end if;
    when arriawriting7 => -- make sure spi is ready again
        status(8)               <= '1';
        if(spi_busy <= '0')then
            spiflashstate           <= arriafifowriting;
        end if;
    when arriafifowriting => -- start writing
        status(9)               <= '1';
        spi_command_arria       <= COMMAND_QUAD_PAGE_PROGRAM;
        spi_addr_arria          <= addr_from_arria;
        spi_continue_arria      <= '1';
        spi_strobe_arria        <= '1';
        wcounter                <= wcounter + 1;
        if ( spiflashfifo_empty = '1' and spi_busy = '0') then
            spiflashstate <= arriawriting8;
            spi_continue_arria      <= '0';
            spi_strobe_arria        <= '0';
            wipseen                 <= '0';
        end if;  
    when arriawriting8 => -- wait for the WIP bit to go off 
        status(10)               <= '1';
        spi_command_arria       <= COMMAND_READ_STATUS_REGISTER1;
        spi_addr_arria          <= (others => '0');
        spi_continue_arria      <= '1';
        spi_strobe_arria        <= '1';

        if(spi_byte_ready = '1' and spi_flash_data_from_flash_int(0) = '1') then
            wipseen <= '1';
        end if;
        if(spi_byte_ready = '1' and spi_flash_data_from_flash_int(0) = '0' and wipseen = '1') then
            spiflashstate           <= idle;
            arriawriting            <= '0';
            spi_continue_arria      <= '0';
            spi_strobe_arria        <= '0'; 
        end if;
            
    when fifowriting =>
        status(11)               <= '1';
        wcounter                 <= wcounter + 1;
        if ( spiflashfifo_empty = '1' ) then
            spiflashstate <= idle;
        end if;    


    when programming =>
        status(12)               <= '1';
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

read_spiflashfifo           <= spi_next_byte;

spi_flash_data_to_flash     <= spiflashfifo_data_out when spiflashstate = fifowriting
                                                       or spiflashstate = arriafifowriting
                              else  spi_flash_data_to_flash_nios;

spi_continue                <= spi_continue_programmer  when spiflashstate = programming
                            else not spiflashfifo_empty when spiflashstate = fifowriting
                            else not spiflashfifo_empty when spiflashstate = arriafifowriting
                            else spi_continue_arria when arriawriting = '1' 
                            else spi_flash_ctrl(1);

spi_strobe                  <= spi_strobe_programmer when spiflashstate = programming
                               else spi_strobe_arria      when arriawriting = '1'
                               else spi_strobe_nios;
spi_command                 <= spi_command_programmer when spiflashstate = programming
                               else spi_command_arria      when arriawriting = '1'
                               else spi_command_nios;
spi_addr                    <= spi_addr_programmer when spiflashstate = programming
                               else spi_addr_arria      when arriawriting = '1'
                               else spi_addr_nios;

spi_flash_status(0)         <= spi_ack;
spi_flash_status(1)         <= spi_next_byte;
spi_flash_status(2)         <= spi_byte_ready;
spi_flash_status(3)         <= spi_busy;

spi_flash_status(5)	        <= spiflashfifo_full;
spi_flash_status(6)         <= spiflashfifo_empty;
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
    spi_byte_out    => spi_flash_data_from_flash_int,
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
    spi_byte_out        => spi_flash_data_from_flash_int,
    spi_byte_ready      => spi_byte_ready,
    spi_flash_request   => spi_flash_request_programmer,
    spi_flash_granted   => spi_flash_granted_programmer,
    --Interface to FPGA
    fpga_conf_done      => fpga_conf_done,
    fpga_nstatus        => fpga_nstatus,
    fpga_nconfig        => fpga_nconfig,
    fpga_data           => fpga_data,
    fpga_clk            => fpga_clk,
    crcerror            => fpp_crcerror,
    timeout             => fpp_timeout,
    debug               => fpp_debug,
    crclocation         => fpp_crclocation --,
);

scfifo_component : altera_mf.altera_mf_components.scfifo
    GENERIC MAP (
            add_ram_output_register => "ON",
            intended_device_family => "Max 10",
            lpm_numwords => 512,
            lpm_showahead => "ON",
            lpm_type => "scfifo",
            lpm_width => 8,
            lpm_widthu => 9,
            overflow_checking => "ON",
            underflow_checking => "ON",
            use_eab => "ON"
    )
    PORT MAP (
            aclr => '0',
            clock => clk100,
            data => spiflashfifo_data_in, 
            rdreq => read_spiflashfifo,
            sclr => not reset_n,
            wrreq => spiflash_to_fifo_we,
            empty => spiflashfifo_empty,
            full  => spiflashfifo_full,
            q => spiflashfifo_data_out
    );

end architecture RTL;