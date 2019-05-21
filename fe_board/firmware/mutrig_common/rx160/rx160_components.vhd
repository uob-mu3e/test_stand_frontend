library ieee;
use ieee.std_logic_1164.all;

package rx160_components is

  component clock_recovery is
    port(
      i_serial_data : in  std_logic;
      i_rst         : in  std_logic;
      i_clk0        : in  std_logic;		   -- two phaseshifted clocks used to
      i_clk90       : in  std_logic;         -- sample the data in 4 time domains

      o_syn_data    : out std_logic 	      -- the data sync to i_clk0
    );
  end component;

  component deserializer is
    port (
      i_ser_clk		: in  std_logic;
      --i_start_val		: in  std_logic_vector(3 downto 0);
      i_rst			   : in  std_logic;
      i_serial_data	: in  std_logic;
      o_byte_clock 	: out std_logic;
      o_byte_data		: out std_logic_vector(7 downto 0);
      o_sync_found	: out std_logic;			-- debug, showing when a syncpacket is found
      o_kontrol		: out std_logic
    );
  end component deserializer;
  
  component dec_8b10b is
    port(
      RESET                          : in  std_logic;	-- Global asynchronous reset (AH) 
      RBYTECLK                       : in  std_logic;	-- Master synchronous receive byte clock
      AI, BI, CI, DI, EI, II         : in  std_logic;
      FI, GI, HI, JI                 : in  std_logic; -- Encoded input (LS..MS)		
      KO                             : out std_logic;	-- Control (K) character indicator (AH)
      HO, GO, FO, EO, DO, CO, BO, AO : out std_logic 	-- Decoded out (MS..LS)
	    );
  end component dec_8b10b; --}}}

  component bclock_gen is
    generic(Nd : Integer := 10);
    port(
      i_clk         : in  std_logic;
      --i_start_val : in std_logic_vector(3 downto 0);
      i_rst			  : in  std_logic;
      o_div_clk     : out std_logic
    );
  end component bclock_gen;
  
  component frame_decomp is
    port (
      i_k_signal 		: in std_logic;
      i_byte_data 	: in std_logic_vector(7 downto 0);
      i_byte_clk 		: in std_logic;
      i_rst 			: in std_logic;
      o_event_data 	: out std_logic_vector(55 downto 0);
      o_event_ready 	: out std_logic;
		o_end_of_frame : out std_logic
    );
  end component frame_decomp;

end package rx160_components;
