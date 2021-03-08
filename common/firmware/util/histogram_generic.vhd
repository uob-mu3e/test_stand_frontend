-----------------------------------------------------------------------------
-- Generic histogram with a simple dual port ram dual clock
-- Date: 08.03.2021
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity histogram_generic is 
generic(
      DATA_WIDTH : natural := 8;
      ADDR_WIDTH : natural := 6
);
   port (
      rclk:         in std_logic;   -- clock for read part
      wclk:         in std_logic;   -- clock for write part
      rst:          in std_logic;   -- all RAM cells are reset to zero, takes a moment
      ena:          in std_logic;   -- if set to '1', and histogram is busy_n = '1', updates histogram
      can_overflow: in std_logic;   -- if set to '1', bins can overflow
      data_in:      in std_logic_vector(DATA_WIDTH-1 downto 0); -- data to be histogrammed
      valid_in:     in std_logic;
      busy_n:       out std_logic;  -- shows that it is ready to accept data
      
      raddr_in:     in std_logic_vector(ADDR_WIDTH-1 downto 0); -- address for readout
      q_out:        out std_logic_vector(DATA_WIDTH-1 downto 0) -- data to be readout, appears 1 cycle after readaddr_in is changed
   );
end histogram_generic;


architecture RTL of histogram_generic is

signal waddr    : std_logic_vector(ADDR_WIDTH-1 downto 0);
signal wdata    : std_logic_vector(DATA_WIDTH-1 downto 0);
signal we       : std_logic;
signal raddr    : std_logic_vector(ADDR_WIDTH-1 downto 0);
signal q        : std_logic_vector(DATA_WIDTH-1 downto 0);


type   state_type is (zeroing, waiting, enabled, readwaiting, writing);
signal state : state_type;

constant addr_all_one : std_logic_vector(waddr'range) := (others => '1');
constant data_all_one : std_logic_vector(q'range) := (others => '1');

signal n_raddr      : natural range 0 to 2**ADDR_WIDTH - 1;
signal n_waddr      : natural range 0 to 2**ADDR_WIDTH - 1;

begin

n_raddr <= to_integer(unsigned(raddr_in));
n_waddr <= to_integer(unsigned(waddr));

i_ram: entity work.true_dual_port_ram_dual_clock
   generic map
   (
      DATA_WIDTH => DATA_WIDTH,
      ADDR_WIDTH => ADDR_WIDTH
   )
   port map
   (
        clk_a   => rclk,    -- towards the outside, here we only read, never write
        clk_b   => wclk,    -- within, we actually need to read and write!
        addr_a  => n_raddr,
        addr_b  => n_waddr,
        data_a  => (others => '0'),
        data_b  => wdata,
        we_a    => '0',
        we_b    => we,
        q_a     => q_out,
        q_b     => q
   );

    -- write state machine
    write_sm: process(wclk)
    begin
    if(rising_edge(wclk)) then
        if(rst = '1')then
            busy_n  <= '0';     -- we are busy after a reset
            state   <= zeroing; -- and we first clear the RAM
            we      <= '0';     -- nothing gets written to the RAM
        else
            case state is
                when zeroing => 
                    we      <= '1';
                    wdata   <= (others => '0');
                    -- coming from reset, we is '0', so we use this to write also address 0!
                    if(we = '0')then
                        waddr   <= (others => '0'); -- reset waddr to 0
                    else
                        waddr   <= waddr + '1';   -- increment
                    end if;
                    if(waddr = addr_all_one) then  -- we will increment address to zero again
                        state   <= waiting;           -- we are done here
                        we      <= '0';               -- don't need to write zeros again
                    end if;
            
                when waiting =>
                    busy_n  <= '1';     -- now we can accept data
                    if (ena = '1')then  -- we are getting the signal to accept data
                        state   <= enabled;
                    end if;
                    
                when enabled =>
                    we     <= '0';
                    if(valid_in = '1')then  -- we get some valid data
                        waddr    <= data_in; -- we set the address, we want to read data!
                        state    <= readwaiting;
                    end if;
                    if(ena = '0')then       -- go back to waiting mode, no more updates to the histogram
                        state   <= waiting;   
                    end if;
                
                when readwaiting =>
                    state   <= writing;
                
                when writing => 
                    we     <= '1';
                    if(can_overflow = '0' and q = data_all_one)then
                        wdata   <= q;
                    else
                        wdata   <= q + '1'; 
                    end if;
                    state   <= enabled;
                    
                when others => 
                    we      <= '0';
                    state   <= waiting;
            end case;
        end if;
    end if;
    end process write_sm;

end RTL;