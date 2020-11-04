-- Testbench for the quad spi IF


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.spiflash_commands.all;


entity spiflash_tb is
end spiflash_tb;

architecture rtl of spiflash_tb is

    component spiflash is
        port (
            reset_n:    in std_logic;
            clk:        in std_logic;
    
            spi_strobe :  in std_logic;
            spi_ack :     out std_logic;
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
    end component;

    signal reset_n:    std_logic;
    signal clk:        std_logic;

    signal spi_strobe :  std_logic;
    signal spi_ack :     std_logic;
    signal spi_command : std_logic_vector(7 downto 0);
    signal spi_addr :    std_logic_vector(23 downto 0);
    signal spi_data :    std_logic_vector(7 downto 0);
    signal spi_next_byte: std_logic;
    signal spi_continue : std_logic;
    signal spi_byte_out : std_logic_vector(7 downto 0);
    signal spi_byte_ready  : std_logic;

    signal spi_sclk:   std_logic;
    signal spi_csn:    std_logic;
    signal spi_mosi:   std_logic;
    signal spi_miso:   std_logic;
    signal spi_D2:     std_logic;
    signal spi_D3:     std_logic;

begin

    dut:spiflash 
        port map(
            reset_n => reset_n,
            clk     => clk,
    
            spi_strobe  => spi_strobe,
            spi_ack     => spi_ack,
            spi_command => spi_command,
            spi_addr    => spi_addr,
            spi_data    => spi_data,
            spi_next_byte  => spi_next_byte,
            spi_continue   => spi_continue,
            spi_byte_out   => spi_byte_out,
            spi_byte_ready => spi_byte_ready,
    
            spi_sclk       => spi_sclk,
            spi_csn        => spi_csn,
            spi_mosi       => spi_mosi,
            spi_miso       => spi_miso,
            spi_D2         => spi_D2,
            spi_D3         => spi_D3
        );

    clkgen:process
    begin
        clk <= '0';
        wait for 5 ns;
        clk <= '1';
        wait for 5 ns;
    end process;

    resetgen: process
    begin
        reset_n <= '0';
        wait for 50 ns;
        reset_n <= '1';
        wait;
    end process;

    stimuli: process    
    begin
        spi_strobe      <= '0';
        spi_command     <= (others => '0');
        spi_addr        <= (others => '0');
        spi_data        <= (others => '0');
        spi_continue    <= '0';
        spi_mosi        <= 'Z';
        wait for 100 ns;
        spi_command     <= COMMAND_WRITE_ENABLE;
        spi_strobe      <= '1';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_WRITE_DISABLE;
        spi_strobe      <= '1';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_READ_STATUS_REGISTER1;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_READ_STATUS_REGISTER2;
        spi_strobe      <= '1';
        spi_miso        <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_READ_STATUS_REGISTER3;
        spi_strobe      <= '1';
        spi_miso        <= '0';
        wait for 220 ns;
        spi_miso        <= '1';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        spi_miso        <= 'Z';
        wait for 15 ns;
        spi_command     <= COMMAND_READ_STATUS_REGISTER1;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_continue    <= '1';
        wait for 400 ns;
        spi_continue    <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_WRITE_ENABLE_VSR;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_READ_DATA;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_addr        <= X"80FF01";
        wait for 700 ns;
        spi_miso        <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_READ_DATA;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_continue    <= '1';
        spi_addr        <= X"80FF01";
        wait for 900 ns;
        spi_miso        <= '0';
        wait for 400 ns;
        spi_miso        <= '1';
        wait for 800 ns;
        spi_miso        <= '0';
        wait for 800 ns;
        spi_continue    <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_FAST_READ;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_addr        <= X"80FF01";
        wait for 900 ns;
        spi_miso        <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_DUAL_OUTPUT_FAST_READ;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_addr        <= X"80FF01";
        wait for 655 ns;
        spi_mosi        <= '0';
        wait for 195 ns;
        spi_miso        <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        spi_mosi        <= 'Z';
        wait for 15 ns;
        spi_command     <= COMMAND_QUAD_OUTPUT_FAST_READ;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_addr        <= X"80FF01";
        wait for 655 ns;
        spi_mosi        <= '0';
        spi_D2          <= '1';
        spi_D3          <= '0';
        wait for 60 ns;
        spi_miso        <= '0';
        spi_D2          <= '0';
        spi_D3          <= '1';
        wait for 60 ns;
        spi_mosi        <= '1';
        spi_D2          <= '1';
        spi_D3          <= '1';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        spi_mosi        <= 'Z';
        spi_D2          <= 'Z';
        spi_D3          <= 'Z';            
        wait for 15 ns;
        spi_command     <= COMMAND_QUAD_OUTPUT_FAST_READ;
        spi_strobe      <= '1';
        spi_miso        <= '1';
        spi_addr        <= X"80FF01";
        spi_continue    <= '1';
        wait for 660 ns;
        spi_mosi        <= '0';
        spi_D2          <= '1';
        spi_D3          <= '0';
        wait for 160 ns;
        spi_miso        <= '0';
        spi_D2          <= '0';
        spi_D3          <= '1';
        wait for 260 ns;
        spi_mosi        <= '1';
        spi_D2          <= '1';
        spi_D3          <= '1';
        wait for 100 ns;
        spi_continue    <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        spi_mosi        <= 'Z';
        spi_D2          <= 'Z';
        spi_D3          <= 'Z';    
        wait for 15 ns;
        spi_command     <= COMMAND_DUAL_IO_FAST_READ;
        spi_strobe      <= '1';
        spi_miso        <= 'Z';
        spi_addr        <= X"80FF01";
        wait for 505 ns;
        spi_mosi        <= '0';
        spi_miso        <= '1';
        wait for 60 ns;
        spi_miso        <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        spi_miso        <= 'Z';
        spi_mosi        <= 'Z';
        spi_D2          <= 'Z';
        spi_D3          <= 'Z';           
        wait for 15 ns;
        spi_command     <= COMMAND_QUAD_IO_FAST_READ;
        spi_strobe      <= '1';
        spi_miso        <= 'Z';
        spi_addr        <= X"80FF01";
        wait for 345 ns;
        spi_mosi        <= '0';
        spi_miso        <= '1';
        spi_D2          <= '1';
        spi_D3          <= '0';
        wait for 20 ns;
        spi_miso        <= '0';
        spi_D2          <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        spi_miso        <= 'Z';
        spi_mosi        <= 'Z';
        spi_D2          <= 'Z';
        spi_D3          <= 'Z';          
        wait for 15 ns;
        spi_command     <= COMMAND_PAGE_PROGRAM;
        spi_strobe      <= '1';
        spi_addr        <= X"80FF01";
        spi_data        <= X"81";
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_PAGE_PROGRAM;
        spi_strobe      <= '1';
        spi_addr        <= X"80FF01";
        spi_data        <= X"81";
        spi_continue    <= '1';
        wait for 680 ns;
        spi_data        <= X"00";
        wait for 160 ns;
        spi_data        <= X"81";
        wait for 160 ns;
        spi_continue    <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_QUAD_PAGE_PROGRAM;
        spi_strobe      <= '1';
        spi_addr        <= X"80FF01";
        spi_data        <= X"81";
        wait until spi_ack = '1';
        spi_strobe      <= '0';
        wait for 15 ns;
        spi_command     <= COMMAND_QUAD_PAGE_PROGRAM;
        spi_strobe      <= '1';
        spi_addr        <= X"80FF01";
        spi_data        <= X"81";
        spi_continue    <= '1';
        wait for 680 ns;
        spi_data        <= X"00";
        wait for 160 ns;
        spi_data        <= X"81";
        wait for 160 ns;
        spi_continue    <= '0';
        wait until spi_ack = '1';
        spi_strobe      <= '0';

        wait;         
        

    end process;    

end rtl;