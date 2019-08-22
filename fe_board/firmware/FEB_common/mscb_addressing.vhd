library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

ENTITY mscb_addressing is
port (
    i_clk           : in  std_logic;
    i_reset         : in  std_logic;
    i_data          : in  std_logic_vector(8 downto 0);
    i_empty         : in  std_logic;
    i_address       : in  std_logic_vector(15 downto 0);
    o_data          : out std_logic_vector(8 downto 0);
    o_wrreq         : out std_logic;
    o_rdreq         : out std_logic
);
END ENTITY;

architecture rtl of mscb_addressing is

    type mscb_addr_state is (idle,rd,proc);
    signal state : mscb_addr_state;

begin

    process(i_clk, i_reset)
    begin
        if i_reset = '1' then
            o_wrreq         <= '0';
            o_rdreq         <= '0';
            o_data          <= (others => '1');
            state           <= idle;
            
        elsif rising_edge(i_clk) then
            
            case state is
                when idle =>
                    o_wrreq         <= '0';
                    if(i_empty = '0') then
                        o_rdreq     <= '1';
                        state       <= rd;
                    else
                        o_rdreq     <= '0';
                    end if;
                    
                when rd =>
                    o_rdreq     <= '0';
                    state       <= proc;
                    
                when proc =>
                    o_wrreq     <='1';
                    o_data      <= i_data;
                    state       <= idle;
                    
                when others =>
                    state       <= idle;
                    
            end case;
        end if;
    end process;
    
end architecture;