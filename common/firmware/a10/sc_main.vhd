----------------------------------------------------------------------------
-- Slow Control Main Unit for Switching Board
--
-- Sebastian Dittmeier, Heidelberg University
-- dittmeier@physi.uni-heidelberg.de
--
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity sc_main is
generic (
    NLINKS : integer := 4
);
port (
    i_clk           : in    std_logic;
    i_reset_n       : in    std_logic;
    i_length_we     : in    std_logic;
    i_length        : in    std_logic_vector(15 downto 0);
    i_mem_data      : in    std_logic_vector(31 downto 0);
    o_mem_addr      : out   std_logic_vector(15 downto 0);
    o_mem_data      : out   work.util.slv32_array_t(NLINKS-1 downto 0);
    o_mem_datak     : out   work.util.slv4_array_t(NLINKS-1 downto 0);
    o_done          : out   std_logic;
    o_state         : out   std_logic_vector(27 downto 0)
);
end entity;

architecture rtl of sc_main is

    --integer to one hot encoding
    function to_onehot(a:integer ; n:integer) return std_logic_vector is
    variable res: std_logic_vector(n-1 downto 0):=(others =>'0');
    begin
            res(a):='1';
            return res;
    end function;

	
	signal addr_reg	: std_logic_vector(15 downto 0) := (others => '0');
	signal wren_reg	: std_logic_vector(15 downto 0);--)(NLINKS-1 downto 0);
	
	type state_type is (idle, read_fpga_id, read_data);
	signal state : state_type;
		
	signal wait_cnt, length_we, length_we_reg : std_logic;
	signal mem_datak : std_logic_vector(3 downto 0); 
	signal mask_addr : std_logic_vector(15 downto 0); 
	signal length : std_logic_vector(15 downto 0);
	signal cur_length : std_logic_vector(15 downto 0);
	
begin

	o_mem_addr <= addr_reg;
	
	process(i_clk, i_reset_n)
	begin
        if ( i_reset_n = '0' ) then
            length_we <= '0';
            length_we_reg <= '0';
        elsif ( rising_edge(i_clk) ) then
            length_we <= '0';
            length_we_reg <= i_length_we;
            if ( i_length_we = '1' and length_we_reg = '0' ) then
                length_we <= '1';
            end if;
        end if;
    end process;

	gen_output:
	for I in 0 to NLINKS-1 generate
		process(i_clk, i_reset_n)
		begin
			if(i_reset_n = '0')then
				o_mem_data(I) <= x"000000BC";
				o_mem_datak(I) <= "0001";
			elsif(rising_edge(i_clk))then
				if (wren_reg(I) = '1') then
					o_mem_data(I) <= i_mem_data;
					o_mem_datak(I) <= mem_datak;
				else
                    o_mem_data(I) <= x"000000BC";
                    o_mem_datak(I) <= "0001";
				end if;
			end if;
		end process;
	end generate gen_output;

	process(i_clk, i_reset_n)
	begin
		if(i_reset_n = '0')then
			addr_reg		<= (others => '0');
			wren_reg		<= (others => '0');
			state			<= idle;
			o_done			<= '0';
			o_state  		<= (others => '0');
			wait_cnt		<= '0';
			mem_datak 		<= (others => '0');
			length 			<= (others => '0');
			cur_length		<= (others => '0');
		elsif(rising_edge(i_clk))then
			wait_cnt	            <= not wait_cnt;
			mem_datak 	            <= (others => '0');
			wren_reg	            <= (others => '0');
			o_state(3 downto 0)	<= x"F";
			
			if (addr_reg = x"FFFF") then
				addr_reg <= (others => '0');
			end if;
			
			case state is
			
				when idle =>
					o_state(3 downto 0) <= x"1";
					o_done				<= '1';
					if(length_we = '1')then
                        state		<= read_fpga_id;
                        length      <= i_length;
                        o_done		<= '0';
                        wait_cnt <= '1';
                    end if;
				
				when read_fpga_id =>
					o_state(3 downto 0) <= x"2";
					if(wait_cnt = '0')then
                        if (i_mem_data(7 downto 0) = x"BC") then
                            state		<= read_data;
                            addr_reg	<= addr_reg + '1';
                            cur_length  <= cur_length + '1';
                            mem_datak <= "0001";
                            mask_addr <= i_mem_data(23 downto 8); -- get fpga id if x"FFFF" write to all links, if 1 first link and so on
                            if(i_mem_data(23 downto 8) = x"FFFF") then
                                wren_reg <= (others => '1');
                            else
                                wren_reg <= i_mem_data(23 downto 8); -- todo fix me to more the 16 addr one hot
                            end if;
                        end if;
					end if;
					
				when read_data =>
					o_state(3 downto 0) <= x"3";
					if(wait_cnt = '0')then
						if(mask_addr(15 downto 0) = x"FFFF") then
							wren_reg <= (others => '1');
						else
							wren_reg <= mask_addr(15 downto 0);
						end if;
						if (length + '1' = cur_length) then
							mem_datak   <= "0001";
							state		<= idle;
							addr_reg	<= (others => '0');
							length	    <= (others => '0');
							cur_length	<= (others => '0');
						else
							cur_length  <= cur_length + '1';
							addr_reg	<= addr_reg + '1';
						end if;
					end if;
										
				when others =>
					state		<= idle;
					addr_reg    <= (others => '0');
					wren_reg	<= (others => '0');
					cur_length	<= (others => '0');
					length	    <= (others => '0');
			end case;
			
		end if;
	end process;

end architecture;
