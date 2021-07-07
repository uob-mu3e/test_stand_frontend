library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.numeric_std.all;


use work.mupix.all;
use work.mudaq.all;


LIBRARY altera_mf;
USE altera_mf.all;

entity sequencer_tb is
end sequencer_tb;

architecture rtl of sequencer_tb is


    COMPONENT scfifo
    GENERIC (
            add_ram_output_register         : STRING;
            almost_full_value               : NATURAL;
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
                    data    : IN sorterfifodata_t;
                    rdreq   : IN STD_LOGIC ;
                    sclr    : IN STD_LOGIC ;
                    wrreq   : IN STD_LOGIC ;
                    almost_full     : OUT STD_LOGIC ;
                    empty   : OUT STD_LOGIC ;
                    q       : OUT sorterfifodata_t
    );
    END COMPONENT;    

constant WRITECLK_PERIOD : time := 10 ns;

signal		reset_n							: std_logic;										-- async reset
signal      reset                           : std_logic;
signal		writeclk						: std_logic;										-- clock for write/input side
signal      tofifo_counters                 : sorterfifodata_t;
signal      fromfifo_counters               : sorterfifodata_t;
signal      read_counterfifo                : std_logic;
signal      write_counterfifo               : std_logic;
signal      counterfifo_almostfull          : std_logic;
signal      counterfifo_empty               : std_logic;

signal      outcommand                      : command_t;
signal      command_enable                  : std_logic;

begin

    reset <= not reset_n;

    scfifo_component : scfifo
    GENERIC MAP (
            add_ram_output_register => "ON",
            almost_full_value => 120,
            intended_device_family => "Arria V",
            lpm_numwords => 128,
            lpm_showahead => "ON",
            lpm_type => "scfifo",
            lpm_width => SORTERFIFORANGE'left + 1,
            lpm_widthu => 7,
            overflow_checking => "ON",
            underflow_checking => "ON",
            use_eab => "ON"
    )
    PORT MAP (
            aclr => '0',
            clock => writeclk,
            data => tofifo_counters,
            rdreq => read_counterfifo,
            sclr => reset,
            wrreq => write_counterfifo,
            almost_full => counterfifo_almostfull,
            empty => counterfifo_empty,
            q => fromfifo_counters
    );




dut: entity work.sequencer_ng
	port map(
		reset_n		=> reset_n,
		clk			=> writeclk,									-- clock
		from_fifo	=> fromfifo_counters,
		fifo_empty	=> counterfifo_empty,
		read_fifo	=> read_counterfifo,
		outcommand	=> outcommand,				
		command_enable	=> command_enable,		
		outoverflow	 => open);				

wclockgen: process
begin
	writeclk	<= '0';
	wait for WRITECLK_PERIOD/2;
	writeclk	<= '1';
	wait for WRITECLK_PERIOD/2;
end process;


resetgen: process
begin
	reset_n <= '0';
	wait for 10 ns;
	reset_n	<= '1';
	wait;
end process;

	
hitgen: process
begin
	tofifo_counters     <= (others => '0');
    write_counterfifo   <= '0';
    wait for 50 ns;
    tofifo_counters(TSBLOCKINFIFORANGE)     <= "0000000";
    write_counterfifo   <= '1';
    wait for WRITECLK_PERIOD;
    tofifo_counters(TSBLOCKINFIFORANGE)     <= "0000000";
    tofifo_counters(0)                      <= '1';
    write_counterfifo   <= '1';
    wait for WRITECLK_PERIOD;
    write_counterfifo   <= '0';
    for i in 0 to 2000 loop
        wait for 9*WRITECLK_PERIOD;
        tofifo_counters(TSBLOCKINFIFORANGE)     <= tofifo_counters(TSBLOCKINFIFORANGE) + '1';
        write_counterfifo   <= '1';
        wait for WRITECLK_PERIOD;
        write_counterfifo   <= '0';
    end loop;
end process;

end architecture rtl;