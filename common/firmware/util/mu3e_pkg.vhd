library ieee;
use ieee.std_logic_1164.all;

-- see Mu3eSpecBook
package mu3e is

    constant PREAMBLE_TYPE_MUPIX_c  : std_logic_vector(5 downto 0) := "111010";
    constant PREAMBLE_TYPE_MUTRIG_c : std_logic_vector(5 downto 0) := "111000";
    constant PREAMBLE_TYPE_SC_c     : std_logic_vector(5 downto 0) := "000111";
    constant PREAMBLE_TYPE_BERT_c   : std_logic_vector(5 downto 0) := "000010";
    constant PREAMBLE_TYPE_IDLE_c   : std_logic_vector(5 downto 0) := "000000";

    -- out of band
    constant SC_OOB_c       : std_logic_vector(1 downto 0) := "00";
    constant SC_READ_c      : std_logic_vector(1 downto 0) := "10";
    constant SC_WRITE_c     : std_logic_vector(1 downto 0) := "11";

    -- start of packet
    constant FIFO_SOP_c     : std_logic_vector(1 downto 0) := "0010";
    -- payload
    constant FIFO_PAYLOAD_c : std_logic_vector(3 downto 0) := "0000";
    -- end of packet
    constant FIFO_EOP_c     : std_logic_vector(3 downto 0) := "0011";
    -- end of run
    constant FIFO_EOR_c     : std_logic_vector(3 downto 0) := "0111";

end package;

package body mu3e is

end package body;
