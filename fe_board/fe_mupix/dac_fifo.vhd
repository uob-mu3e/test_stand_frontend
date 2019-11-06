library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dac_fifo is
port (
   
   	-- mupix dac regs
	i_reg_add               : in std_logic_vector(7 downto 0);
	i_reg_re                : in std_logic;
	o_reg_rdata       		: out std_logic_vector(31 downto 0);
	i_reg_we   				: in std_logic;
	i_reg_wdata 		    : in std_logic_vector(31 downto 0);

    -- mupix board dac data
    o_board_dac_data  : out std_logic_vector(31 downto 0);
    i_board_dac_ren   : in std_logic;
    o_board_dac_fifo_empty : out std_logic;
    o_board_dac_ready : out std_logic;

    i_board_dac_data  : in std_logic_vector(31 downto 0);
    i_board_dac_we  : in std_logic;


    -- mupix chip dac data
    o_chip_dac_data  : out std_logic_vector(31 downto 0);
    i_chip_dac_ren   : in std_logic;
    o_chip_dac_fifo_empty : out std_logic;
    o_chip_dac_ready : out std_logic;

    i_chip_dac_data : in std_logic_vector(31 downto 0);
    i_chip_dac_we  : in std_logic;
    
    i_reset_n         : in std_logic;
    -- 156.25 MHz
    i_clk           : in std_logic--;
);
end entity;

architecture arch of dac_fifo is

    signal reset : std_logic;

    signal board_dac_data : std_logic_vector(31 downto 0);
    signal chip_dac_data : std_logic_vector(31 downto 0);
    signal board_dac_we : std_logic;
    signal chip_dac_we : std_logic;
    signal board_dac_re : std_logic;
    signal chip_dac_re : std_logic;
    

begin

    reset <= not i_reset_n;

    handle_fifo : process (i_clk, i_reset_n)
    begin 
        if (i_reset_n = '0') then 
            board_dac_data      <= (others => '0');
            chip_dac_data       <= (others => '0');
            board_dac_we        <= '0';
            chip_dac_we         <= '0';
            o_board_dac_ready   <= '0';
            o_chip_dac_ready    <= '0';
            board_dac_re        <= '0';
            chip_dac_re         <= '0';
        elsif rising_edge(i_clk) then 
            board_dac_we        <= '0';
            chip_dac_we         <= '0';
            o_board_dac_ready   <= '0';
            o_chip_dac_ready    <= '0';
            board_dac_re        <= '0';
            chip_dac_re         <= '0';
            if ( i_reg_add = x"81" and i_reg_we = '1' ) then
                board_dac_data  <= i_reg_wdata;
                board_dac_we    <= '1';
            end if;
            if ( i_reg_add = x"82" and i_reg_we = '1' ) then
                chip_dac_data  <= i_reg_wdata;
                chip_dac_we    <= '1';
            end if;
            if ( i_reg_add = x"80" and i_reg_we = '1' and i_reg_wdata = x"AAAAAAAA" ) then
                o_board_dac_ready <= '1';
            end if;
            if ( i_reg_add = x"80" and i_reg_we = '1' and i_reg_wdata = x"BBBBBBBB" ) then
                o_chip_dac_ready <= '1';
            end if;
            if ( i_reg_add = x"83" and i_reg_re = '1' ) then
                board_dac_re <= '1';
            end if;
            if ( i_reg_add = x"84" and i_reg_re = '1' ) then
                chip_dac_re <= '1';
            end if;
        end if;
    end process handle_fifo;

    board_dac_fifo_write : work.ip_scfifo
    generic map (
        ADDR_WIDTH => 4,
        DATA_WIDTH => 32--,
    )
    port map (
        empty           => o_board_dac_fifo_empty,
        rdreq           => i_board_dac_ren,
        q               => o_board_dac_data,

        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        
        full            => open,
        wrreq           => board_dac_we,
        data            => board_dac_data,

        sclr            => reset,
        clock           => i_clk--,
    );
    
    board_dac_fifo_read : work.ip_scfifo
    generic map (
        ADDR_WIDTH => 4,
        DATA_WIDTH => 32--,
    )
    port map (
        empty           => open,
        rdreq           => board_dac_re,
        q               => o_reg_rdata,

        almost_empty    => open,
        almost_full     => open,
        usedw           => open,
        
        full            => open,
        wrreq           => i_board_dac_we,
        data            => i_board_dac_data,

        sclr            => reset,
        clock           => i_clk--,
    );

--    chip_dac_fifo : component fifo
--    port map ( 
--        DataIn     => chip_dac_data,
--        WriteEn    => chip_dac_we,
--        ReadEn     => i_chip_dac_ren,
--        CLK        => i_clk,
--        DataOut    => o_chip_dac_data,
--        Full       => open,
--        Empty      => o_chip_dac_fifo_empty,
--        RST        => reset--,
--    );
--
--    chip_dac_fifo_read : component fifo
--    port map ( 
--        DataIn     => i_chip_dac_data,
--        WriteEn    => i_chip_dac_we,
--        ReadEn     => chip_dac_re,
--        CLK        => i_clk,
--        DataOut    => o_reg_rdata,
--        Full       => open,
--        Empty      => open,
--        RST        => reset--,
--    );

end architecture;
