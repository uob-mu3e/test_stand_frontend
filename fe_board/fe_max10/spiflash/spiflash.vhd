-- Quad SPI flash interface
-- Niklaus Berger niberger@uni-mainz.de
-- August 2020


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


entity spiflash is
    port (
        reset_n:    in std_logic;
        clk:        in std_logic;

        spi_strobe :  in std_logic;
        spi_ack :     out std_logic;
		  spi_busy :	 out std_logic;
        spi_command : in std_logic_vector(7 downto 0);
        spi_addr :    in std_logic_vector(23 downto 0);
        spi_data :    in std_logic_vector(7 downto 0);
        spi_next_byte: out std_logic;
        spi_continue : in std_logic;
        spi_byte_out : out std_logic_vector(7 downto 0);
        spi_byte_ready  : out std_logic;

        spi_sclk:   out std_logic;
        spi_csn:    out std_logic;
        spi_mosi:   inout std_logic;
        spi_miso:   inout std_logic;
        spi_D2:     inout std_logic;
        spi_D3:     inout std_logic
    );
end spiflash;

architecture rtl of spiflash is


    type spi_state_type is (idle, command, address, dual_address, quad_address, read, dummy_read, 
                            dual_read, quad_read, quad_read_fast, write, quad_write, stop, waits);
    signal spi_state: spi_state_type;

    signal strobe_last: std_logic;

    signal shiftreg : std_logic_vector(23 downto 0);
    signal dualshiftreg : std_logic_vector(31 downto 0);
    signal quadshiftreg : std_logic_vector(31 downto 0);    
    signal count : natural range 0 to 63;
    signal toggle : std_logic;
    signal readbyteshiftreg :  std_logic_vector(7 downto 0);
    signal dualreadbyteshiftreg :  std_logic_vector(7 downto 0); 
    signal quadreadbyteshiftreg :  std_logic_vector(7 downto 0);       
    signal dummyread : std_logic;
    signal writeshiftreg : std_logic_vector(7 downto 0);
    signal quadwriteshiftreg : std_logic_vector(7 downto 0);

begin


process(clk, reset_n)
begin
if(reset_n = '0') then
    spi_sclk <= '0';
    spi_csn  <= '1';
    spi_mosi <= 'Z';
    spi_miso <= 'Z';
    spi_D2   <= 'Z';
    spi_D3   <= 'Z';
    spi_state <= idle;
    strobe_last <= '0';

    spi_next_byte  <= '0';
    spi_ack   <= '0';
	 spi_busy	<= '0';
    spi_byte_ready <= '0';

elsif(clk'event and clk = '1') then
    toggle <= not toggle;
    spi_ack   <= '0';
    spi_byte_ready <= '0';
    spi_next_byte  <= '0';

    strobe_last <= spi_strobe;

    case (spi_state) is
    when idle =>
        if(spi_strobe = '1' and strobe_last = '0')then            
            spi_csn <= '0';
            spi_state <= command;
            shiftreg(7 downto 0) <= spi_command;
            count <= 0;
            toggle <= '0';
				spi_busy	<= '1';
        end if;
    when command =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            spi_mosi <= shiftreg(7);
            shiftreg(23 downto 1) <= shiftreg(22 downto 0);
        else
            spi_sclk <= '1';
            count <= count + 1;
        end if;
        if(count = 8)then
            if(spi_command = work.util.COMMAND_WRITE_ENABLE or 
               spi_command = work.util.COMMAND_WRITE_DISABLE or
               spi_command = work.util.COMMAND_WRITE_ENABLE_VSR or
               spi_command = work.util.COMMAND_CHIP_ERASE or
               spi_command = work.util.COMMAND_ENABLE_RESET or
               spi_command = work.util.COMMAND_RESET or
               spi_command = work.util.COMMAND_ERASE_SECURITY_REGISTERS)  then

                spi_state <= stop;
                toggle <= '0';

            elsif(spi_command = work.util.COMMAND_READ_DATA or
               spi_command = work.util.COMMAND_FAST_READ or
               spi_command = work.util.COMMAND_DUAL_OUTPUT_FAST_READ or
               spi_command = work.util.COMMAND_QUAD_OUTPUT_FAST_READ or
               spi_command = work.util.COMMAND_PAGE_PROGRAM or
               spi_command = work.util.COMMAND_QUAD_PAGE_PROGRAM or
               spi_command = work.util.COMMAND_FAST_PAGE_PROGRAM or
               spi_command = work.util.COMMAND_SECTOR_ERASE or
               spi_command = work.util.COMMAND_BLOCK_ERASE_32 or
               spi_command = work.util.COMMAND_BLOCK_ERASE_64 or
               spi_command = work.util.COMMAND_PROG_SECURITY_REGISTERS or
               spi_command = work.util.COMMAND_READ_SECURITY_REGISTERS) then

                spi_state <= address;
                count     <= 0;
                toggle    <= '1';
                shiftreg  <= spi_addr;
                spi_mosi  <= spi_addr(23);

            elsif(spi_command = work.util.COMMAND_DUAL_IO_FAST_READ
                 ) then

                spi_state <= dual_address;
                count     <= 0;
                toggle    <= '1';
                dualshiftreg(31 downto 8)  <= spi_addr;    
                dualshiftreg(7 downto 0)  <= (others => '0');
                spi_mosi    <= spi_addr(22);
                spi_miso    <= spi_addr(23);

            elsif(spi_command = work.util.COMMAND_QUAD_IO_FAST_READ or
                  spi_command = work.util.COMMAND_QUAD_IO_WORD_FAST_READ
            ) then

                spi_state <= quad_address;
                count     <= 0;
                toggle    <= '1';
                quadshiftreg(31 downto 8)  <= spi_addr;    
                quadshiftreg(7 downto 0)  <= (others => '0');
                spi_mosi    <= spi_addr(20);
                spi_miso    <= spi_addr(21);
                spi_D2      <= spi_addr(22);
                spi_D3      <= spi_addr(23);

            elsif(spi_command = work.util.COMMAND_READ_STATUS_REGISTER1 or
                  spi_command = work.util.COMMAND_READ_STATUS_REGISTER2 or
                  spi_command = work.util.COMMAND_READ_STATUS_REGISTER3 or
                  spi_command = work.util.COMMAND_JEDEC_ID) then

                spi_state <= read;
                dummyread <= '0';
                count     <= 0;
                toggle    <= '1';
                spi_miso  <= 'Z';


            elsif(spi_command = work.util.COMMAND_WRITE_STATUS_REGISTER1 or
                  spi_command = work.util.COMMAND_WRITE_STATUS_REGISTER2 or
                  spi_command = work.util.COMMAND_WRITE_STATUS_REGISTER3) then

                spi_state <= write;
                count     <= 0;
                toggle    <= '0';
                spi_miso  <= 'Z';

                writeshiftreg <= spi_data;

            end if;
        end if;

    when address =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            spi_mosi <= shiftreg(22);
            shiftreg(23 downto 1) <= shiftreg(22 downto 0);
        else
            spi_sclk <= '1';
            count <= count + 1;
        end if;
        if(count = 24)then
            if(spi_command = work.util.COMMAND_READ_DATA or
              spi_command = work.util.COMMAND_FAST_READ or
              spi_command = work.util.COMMAND_READ_SECURITY_REGISTERS
            ) then

                spi_state <= read;
                count     <= 0;
                toggle    <= '1';
                spi_miso  <= 'Z';
                if(spi_command = work.util.COMMAND_FAST_READ or
                   spi_command = work.util.COMMAND_READ_SECURITY_REGISTERS
                ) then
                    dummyread <= '1';
                else 
                    dummyread <= '0';
                end if;

            elsif(spi_command = work.util.COMMAND_DUAL_OUTPUT_FAST_READ) then
                
                spi_state <= dual_read;
                count     <= 0;
                toggle    <= '1';
                spi_miso  <= 'Z';
                spi_mosi  <= 'Z';
                dummyread <= '1';

            elsif(spi_command = work.util.COMMAND_QUAD_OUTPUT_FAST_READ) then
                
                spi_state <= quad_read_fast;
                count     <= 0;
                toggle    <= '1';
                spi_miso  <= 'Z';
                spi_mosi  <= 'Z';
                spi_D2    <= 'Z';
                spi_D3    <= 'Z';
                dummyread <= '1';   
                
            elsif(spi_command = work.util.COMMAND_PAGE_PROGRAM or
               spi_command = work.util.COMMAND_FAST_PAGE_PROGRAM or
               spi_command = work.util.COMMAND_PROG_SECURITY_REGISTERS
            ) then
                spi_state <= write;
                count     <= 0;
                toggle    <= '1';  
                writeshiftreg <= spi_data;
                spi_next_byte      <= '1';
                spi_mosi  <= spi_data(7);

            elsif(spi_command = work.util.COMMAND_QUAD_PAGE_PROGRAM
            ) then    
                spi_state <= quad_write;
                count     <= 0;
                toggle    <= '1';   
                quadwriteshiftreg <= spi_data;
                spi_next_byte      <= '1';
                spi_mosi    <= spi_data(4);
                spi_miso    <= spi_data(5);
                spi_D2      <= spi_data(6);
                spi_D3      <= spi_data(7);
                
            elsif(spi_command = work.util.COMMAND_SECTOR_ERASE or
                  spi_command = work.util.COMMAND_BLOCK_ERASE_32 or
                  spi_command = work.util.COMMAND_BLOCK_ERASE_64
                  ) then
                
                spi_state <= stop;
                toggle <= '0';  

            end if;
        end if;

    when dual_address =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            spi_mosi <= dualshiftreg(28);
            spi_miso <= dualshiftreg(29);
            dualshiftreg(31 downto 2) <= dualshiftreg(29 downto 0);
            count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 16)then    
            spi_state <= dual_read;
            count     <= 0;
            toggle    <= '0';
            spi_miso  <= 'Z';
            spi_mosi  <= 'Z';
            dummyread <= '0';
        end if;
    when quad_address =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            spi_mosi <= quadshiftreg(24);
            spi_miso <= quadshiftreg(25);
            spi_D2   <= quadshiftreg(26);
            spi_D3   <= quadshiftreg(27);
            quadshiftreg(31 downto 4) <= quadshiftreg(27 downto 0);
            count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 8)then    
            spi_state <= quad_read;
            count     <= 0;
            toggle    <= '0';
            spi_miso  <= 'Z';
            spi_mosi  <= 'Z';
            spi_D2    <= 'Z';
            spi_D3    <= 'Z';
            dummyread <= '1';
        end if;    
     
    -- Note that all the read states generate one extra SPI clock cycle in the end   

    when read =>
        if  (toggle = '0') then
            spi_sclk <= '0';
				readbyteshiftreg(0) <= spi_miso;
            readbyteshiftreg(7 downto 1) <= readbyteshiftreg(6 downto 0);
				count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 8) then
            spi_byte_out <= readbyteshiftreg;
            spi_byte_ready <= not dummyread;
            dummyread <= '0';
            if(spi_continue = '1' or dummyread = '1')then
                count <= 0;
            else
                spi_sclk <= '0';
                spi_state <= stop;
                toggle <= '0'; 
            end if;
        end if;
        
    when dual_read =>
        if  (toggle = '0') then
            spi_sclk <= '0';
		    dualreadbyteshiftreg(0) <= spi_mosi;
            dualreadbyteshiftreg(1) <= spi_miso;
            dualreadbyteshiftreg(7 downto 2) <= dualreadbyteshiftreg(5 downto 0);
            count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 4 or count = 8) then
            spi_byte_out <= dualreadbyteshiftreg;
            spi_byte_ready <= not dummyread;
            if(dummyread = '1') then
                if(count = 8) then
                    count <= 0;
                    dummyread <= '0';
                end if;    
            elsif(spi_continue = '1' )then
                count <= 0;
            else
                spi_sclk <= '0';
                spi_state <= stop;
                toggle <= '0'; 
            end if;
        end if;   

    when quad_read =>
        if  (toggle = '0') then
            spi_sclk <= '0';
				quadreadbyteshiftreg(0) <= spi_mosi;
            quadreadbyteshiftreg(1) <= spi_miso;
            quadreadbyteshiftreg(2) <= spi_D2;
            quadreadbyteshiftreg(3) <= spi_D3;
            quadreadbyteshiftreg(7 downto 4) <= quadreadbyteshiftreg(3 downto 0);
            count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 2 or count = 4) then
            spi_byte_out <= quadreadbyteshiftreg;
            spi_byte_ready <= not dummyread;
            if(dummyread = '1') then
                if(count = 4) then
                    count <= 0;
                    dummyread <= '0';
                end if;    
            elsif(spi_continue = '1' )then
                count <= 0;
            else
                spi_sclk <= '0';
                spi_state <= stop;
                toggle <= '0'; 
            end if;
        end if;       
   when quad_read_fast =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            count <= count + 1;
        else
			quadreadbyteshiftreg(0) <= spi_mosi;
            quadreadbyteshiftreg(1) <= spi_miso;
            quadreadbyteshiftreg(2) <= spi_D2;
            quadreadbyteshiftreg(3) <= spi_D3;
            quadreadbyteshiftreg(7 downto 4) <= quadreadbyteshiftreg(3 downto 0);
            spi_sclk <= '1';
        end if;
        if(count = 2 or count = 8) then
            spi_byte_out <= quadreadbyteshiftreg;
            spi_byte_ready <= not dummyread;
            if(dummyread = '1') then
                if(count = 8) then
                    count <= 0;
                    dummyread <= '0';
                end if;    
            elsif(spi_continue = '1' )then
                count <= 0;
            else
                spi_sclk <= '0';
                spi_state <= stop;
                toggle <= '0'; 
            end if;
        end if;  
    when write =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            spi_mosi <= writeshiftreg(6);
            writeshiftreg(7 downto 1) <= writeshiftreg(6 downto 0);
            count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 7 and toggle = '0')then
            if(spi_continue = '1') then
                writeshiftreg <= spi_data;
                spi_next_byte <= '1';
                spi_mosi <= spi_data(7);
                count <= 0;
            else
                spi_sclk <= '0';
                spi_state <= stop;
                toggle <= '0'; 
            end if;    
        end if;

    when quad_write =>
        if  (toggle = '0') then
            spi_sclk <= '0';
            spi_mosi <= quadwriteshiftreg(0);
            spi_miso <= quadwriteshiftreg(1);
            spi_D2   <= quadwriteshiftreg(2);
            spi_D3   <= quadwriteshiftreg(3);
            quadwriteshiftreg(7 downto 4) <= quadwriteshiftreg(3 downto 0);
            count <= count + 1;
        else
            spi_sclk <= '1';
        end if;
        if(count = 1 and toggle = '0')then
            if(spi_continue = '1') then
                quadwriteshiftreg <= spi_data;
                spi_next_byte <= '1';
                spi_mosi <= spi_data(4);
                spi_miso <= spi_data(5);
                spi_D2   <= spi_data(6);
                spi_D3   <= spi_data(7);
                count    <= 0;
            else
                spi_sclk <= '0';
                spi_state <= stop;
                toggle <= '0'; 
            end if;    
        end if;    

    when stop =>        
        spi_sclk <= '0';
        spi_mosi <= 'Z';
        spi_miso <= 'Z';
        spi_D2   <= 'Z';
        spi_D3   <= 'Z';

        if(toggle = '1')then
            spi_csn <= '1';
            spi_state <= waits;
            toggle <= '0';
        end if;

    when waits =>
        if(toggle = '1')then      
            spi_ack     <= '1';
            spi_state   <= idle;
				spi_busy		<= '0';
        end if;  
    when others =>
        spi_state <= idle;
        spi_csn <= '1';
        spi_sclk <= '0';
        spi_mosi <= 'Z';
        spi_miso <= 'Z';
        spi_D2   <= 'Z';
        spi_D3   <= 'Z';
    end case;
end if;
end process;

end architecture rtl;