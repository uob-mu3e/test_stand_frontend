-- Testbench for the FPGA programming IF


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.spiflash_commands.all;


entity programming_tb is
end programming_tb;

architecture rtl of programming_tb is

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

    component ps_programmer is
        port (
            reset_n                 : in std_logic;
            clk                     : in std_logic;
    
            start                   : in std_logic;
            start_address           : in std_logic_vector(23 downto 0) := X"000000";
    
            -- Interface to SPI flash
            spi_strobe              : out std_logic;
            spi_command             : out std_logic_vector(7 downto 0);
            spi_addr                : out std_logic_vector(23 downto 0);
            spi_next_byte           : in std_logic;
            spi_continue            : out std_logic;
            spi_byte_out            : in std_logic_vector(7 downto 0);
            spi_byte_ready          : in std_logic;
    
            spi_flash_request       : out std_logic;
            spi_flash_granted       : in std_logic;
    
            -- Interface to FPGA
            fpga_conf_done		    : in    std_logic;
            fpga_nstatus			: in 	std_logic;
            fpga_nconfig			: out   std_logic;
            fpga_data				: out	std_logic_vector(7 downto 0);
            fpga_clk				: out	std_logic
        );
    end component;

    signal reset_n:    std_logic;
    signal clk:        std_logic;
    signal slowclk:    std_logic;

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

    signal start : std_logic;
    signal spi_flash_request : std_logic;
    signal spi_flash_granted : std_logic;

    signal fpga_conf_done		:     std_logic;
    signal fpga_nstatus			:  	  std_logic;
    signal fpga_nconfig			:     std_logic;
    signal fpga_data		    :     std_logic_vector(7 downto 0);
    signal fpga_clk				:     std_logic;

    signal lfsr                 :     std_logic_vector(7 downto 0) := X"01";

begin

    dut1:spiflash 
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



    dut2:ps_programmer
        port map(
            reset_n    => reset_n,
            clk        => clk,
            start      => start,
            start_address => X"88BB11",
        
            -- Interface to SPI flash
            spi_strobe  => spi_strobe,
            spi_command => spi_command,
            spi_addr    => spi_addr,
            spi_next_byte    => spi_next_byte,
            spi_continue     => spi_continue,
            spi_byte_out     => spi_byte_out,
            spi_byte_ready   => spi_byte_ready,
        
            spi_flash_request       => spi_flash_request,
            spi_flash_granted       => spi_flash_granted,
        
            -- Interface to FPGA
            fpga_conf_done		    => fpga_conf_done,
            fpga_nstatus			=> fpga_nstatus,
            fpga_nconfig			=> fpga_nconfig,
            fpga_data				=> fpga_data,
            fpga_clk				=> fpga_clk
        );

    spi_flash_granted <= spi_flash_request after 50 ns;
    fpga_nstatus      <= fpga_nconfig after 700 ns;
    

    clkgen:process
    begin
        clk <= '0';
        wait for 5 ns;
        clk <= '1';
        wait for 5 ns;
    end process;

    slowclkgen:process
    begin
    slowclk <= '0';
    wait for 10 ns;
    slowclk <= '1';
    wait for 10 ns;
    end process;

    resetgen: process
    begin
        reset_n <= '0';
        wait for 50 ns;
        reset_n <= '1';
        wait;
    end process;

    startgen: process
    begin
    start <= '0';
    wait for 200 ns;
    start <= '1';
    wait for 10 ns;
    start <= '0';
    wait;
    end process;

    confdonegen:process
    begin
    fpga_conf_done <= '0';
    wait for 8010 ns;
    fpga_conf_done <= '1';
    wait;
    end process;

    PROCESS(slowclk)
        variable tmp : STD_LOGIC := '0';
    BEGIN
        IF rising_edge(slowclk) THEN
            tmp := lfsr(4) XOR lfsr(3) XOR lfsr(2) XOR lfsr(0);
            lfsr <= tmp & lfsr(7 downto 1);
        END IF;
    END PROCESS;

    spi_miso <= lfsr(7);

end rtl;