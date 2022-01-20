--

library ieee;
use ieee.std_logic_1164.all;

entity ip_scfifo_v2 is
generic (
    g_ADDR_WIDTH : positive := 8;
    g_DATA_WIDTH : positive := 8;
    g_RREG_N : natural := 0;
    g_WREG_N : natural := 0;
    g_SHOWAHEAD : string := "ON";
    g_DEVICE_FAMILY : string := "Arria 10"--;
);
port (
    o_usedw         : out   std_logic_vector(g_ADDR_WIDTH-1 downto 0);

    o_rdata         : out   std_logic_vector(g_DATA_WIDTH-1 downto 0);
    i_rack          : in    std_logic; -- read enable (request, acknowledge)
    o_rempty        : out   std_logic;
    o_almost_empty  : out   std_logic;

    i_wdata         : in    std_logic_vector(g_DATA_WIDTH-1 downto 0);
    i_we            : in    std_logic; -- write enable (request)
    o_wfull         : out   std_logic;
    o_almost_full   : out   std_logic;

    i_clk           : in    std_logic;
    i_reset_n       : in    std_logic--;
);
end entity;

library altera_mf;
use altera_mf.altera_mf_components.all;

architecture arch of ip_scfifo_v2 is

    signal fifo_rdata : std_logic_vector(o_rdata'range);
    signal fifo_wdata : std_logic_vector(i_wdata'range);
    signal fifo_rack, fifo_rempty, rreset_n, fifo_we, fifo_wfull, wreset_n : std_logic;

begin

    assert ( true
        and ( g_RREG_N = 0 or g_SHOWAHEAD = "ON" )
    ) report "ip_dcfifo_v2"
        & ", ADDR_WIDTH = " & integer'image(g_ADDR_WIDTH)
        & ", DATA_WIDTH = " & integer'image(o_rdata'length)
    severity failure;

    scfifo_component : scfifo
    GENERIC MAP (
        lpm_type => "scfifo",
        lpm_widthu => g_ADDR_WIDTH,
        lpm_numwords => 2**g_ADDR_WIDTH,
        almost_empty_value => 2**(g_ADDR_WIDTH/2),
        almost_full_value => 2**g_ADDR_WIDTH - 2**(g_ADDR_WIDTH/2),
        lpm_width => g_DATA_WIDTH,
        lpm_showahead => g_SHOWAHEAD,
        add_ram_output_register => "OFF",
        use_eab => "ON",
        overflow_checking => "ON",
        underflow_checking => "ON",
        intended_device_family => g_DEVICE_FAMILY--,
    )
    PORT MAP (
        usedw => o_usedw,

        q => fifo_rdata,
        rdreq => fifo_rack,
        empty => fifo_rempty,
        almost_empty => o_almost_empty,

        data => fifo_wdata,
        wrreq => fifo_we,
        full => fifo_wfull,
        almost_full => o_almost_full,

        clock => i_clk,
        sclr => not i_reset_n--,
    );

    generate_rreg_0 : if ( g_RREG_N = 0 or g_SHOWAHEAD /= "ON" ) generate
        o_rdata <= fifo_rdata;
        fifo_rack <= i_rack;
        o_rempty <= fifo_rempty;
    end generate;

    generate_rreg : if ( g_RREG_N > 0 and g_SHOWAHEAD = "ON" ) generate
        -- read through reg fifo
        e_fifo_rreg : entity work.fifo_reg
        generic map (
            g_DATA_WIDTH => o_rdata'length,
            g_N => g_RREG_N--,
        )
        port map (
            o_rdata => o_rdata,
            i_rack => i_rack,
            o_rempty => o_rempty,

            i_wdata => fifo_rdata,
            i_we => not fifo_rempty,
            o_wfull_n => fifo_rack,

            i_reset_n => rreset_n,
            i_clk => i_clk--,
        );
    end generate;

    generate_wreg_0 : if ( g_WREG_N = 0 ) generate
        fifo_wdata <= i_wdata;
        fifo_we <= i_we;
        o_wfull <= fifo_wfull;
    end generate;

    generate_wreg : if ( g_WREG_N > 0 ) generate
        -- write through reg fifo
        e_fifo_wreg : entity work.fifo_reg
        generic map (
            g_DATA_WIDTH => i_wdata'length,
            g_N => g_WREG_N--,
        )
        port map (
            o_rdata => fifo_wdata,
            i_rack => not fifo_wfull,
            o_rempty_n => fifo_we,

            i_wdata => i_wdata,
            i_we => i_we,
            o_wfull => o_wfull,

            i_reset_n => wreset_n,
            i_clk => i_clk--,
        );
    end generate;

end architecture;
