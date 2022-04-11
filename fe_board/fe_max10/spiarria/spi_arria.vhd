library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spi_arria is
port(
    ------ SPI
    i_SPI_csn   : in    std_logic;
    i_SPI_clk   : in    std_logic;
    io_SPI_mosi : inout std_logic;
    io_SPI_miso : inout std_logic;
    io_SPI_D1   : inout std_logic;
    io_SPI_D2   : inout std_logic;
    io_SPI_D3   : inout std_logic;

    clk100       : in std_logic;
    reset_n     : in std_logic;
    addr        : out std_logic_vector(6 downto 0);
    addroffset  : out std_logic_vector(7 downto 0);
    rw          : out std_logic;
    data_to_arria   : in std_logic_vector(31 downto 0);
    next_data       : out std_logic;
    word_from_arria : out std_logic_vector(31 downto 0);
    word_en         : out std_logic;
    byte_from_arria : out std_logic_vector(7 downto 0);
    byte_en         : out std_logic
);
end entity;


architecture RTL of spi_arria is

    type spistate_type is (idle, address, writing, waiting, reading);
    signal spistate : spistate_type;
    signal addrshiftregister : std_logic_vector(7 downto 0);
    signal datashiftregister : std_logic_vector(31 downto 0);
    signal datareadshiftregister : std_logic_vector(31 downto 0);
    signal clklast: std_logic;
    signal nibblecount : integer;
    signal addroffset_int : integer;
    signal haveread : std_logic;

    signal csn_reg   : std_logic;
    signal clk_reg 	: std_logic;
    signal mosi_reg	: std_logic;
    signal miso_reg	: std_logic;
    signal d1_reg		: std_logic;
    signal d2_reg		: std_logic;
    signal d3_reg		: std_logic;

begin

    addroffset <= std_logic_vector(to_unsigned(addroffset_int, addroffset'length));

    process(clk100, reset_n)
    begin
    if(reset_n = '0')then
        io_SPI_mosi     <= 'Z';
        io_SPI_miso     <= 'Z';
        io_SPI_D1       <= 'Z';
        io_SPI_D2       <= 'Z';
        io_SPI_D3       <= 'Z';
        spistate        <= idle;
        word_en         <= '0';
        byte_en         <= '0';
        next_data       <= '0';
        clklast         <= '0';
        clk_reg			  <= '0';
        csn_reg			  <= '1';
    elsif(clk100'event and clk100 = '1')then
        word_en         <= '0';
        byte_en         <= '0';
        next_data       <= '0';

		  csn_reg			<= i_SPI_csn;
		  clk_reg			<= i_SPI_clk;
		  miso_reg			<= io_SPI_miso;
		  mosi_reg			<= io_SPI_mosi;
		  d1_reg				<= io_SPI_D1;
		  d2_reg				<= io_SPI_D2;
		  d3_reg				<= io_SPI_D3;

        clklast         <= clk_reg;

        case spistate is
        when idle =>
            io_SPI_mosi     <= 'Z';
            io_SPI_miso     <= 'Z';
            io_SPI_D1       <= 'Z';
            io_SPI_D2       <= 'Z';
            io_SPI_D3       <= 'Z';
				nibblecount 	<=  0;
            if(csn_reg = '0') then
                spistate    <= address;
                nibblecount <= 0;
                addroffset_int  <= 0;
            end if;
        when address =>
            if(clklast = '0' and clk_reg = '1') then
                addrshiftregister(4) <=  mosi_reg;
                addrshiftregister(5) <=  d1_reg;
                addrshiftregister(6) <=  d2_reg;
                addrshiftregister(7) <=  d3_reg;
                addrshiftregister(3 downto 0) <= addrshiftregister(7 downto 4);
                nibblecount     <= nibblecount +1;
            end if;
            if(nibblecount > 1) then
               addr <=    addrshiftregister(6 downto 0);
               rw   <=    addrshiftregister(7);
               if(addrshiftregister(7) = '1') then
                    spistate    <= writing;
                    nibblecount <= 0;
               else
                    spistate    <= waiting;
                    nibblecount <= 0;
               end if;
            end if;

				if(csn_reg = '1') then
                spistate <= idle;
            end if;

        when writing =>
            haveread <= '0';
            if(clklast = '0' and clk_reg = '1') then
                datashiftregister(28) <=  mosi_reg;
                datashiftregister(29) <=  d1_reg;
                datashiftregister(30) <=  d2_reg;
                datashiftregister(31) <=  d3_reg;
                datashiftregister(27 downto 0) <= datashiftregister(31 downto 4);
                nibblecount     <= nibblecount +1;
                haveread        <= '1';
            end if;
            if(haveread = '1' and nibblecount mod 2 = 0 and nibblecount > 0)then
                byte_from_arria     <= datashiftregister(31 downto 24);
                byte_en             <= '1';
            end if;
            if(haveread = '1' and nibblecount mod 8 = 1 and nibblecount > 1)then
                addroffset_int          <= addroffset_int + 1;
            end if;
            if(haveread = '1' and nibblecount mod 8 = 0 and nibblecount > 0)then
                word_from_arria   <= datashiftregister;
                word_en         <= '1';
            end if;

            if(csn_reg = '1') then
                spistate <= idle;
            end if;
        when waiting =>
            if(clklast = '0' and clk_reg = '1') then
                nibblecount     <= nibblecount +1;
            end if;
            if(nibblecount = 1) then
                spistate    <= reading;
                nibblecount <= 0;
                datareadshiftregister   <= data_to_arria;
                next_data         <= '1';
            end if;

            if(csn_reg = '1') then
                spistate <= idle;
            end if;
        when reading =>
            haveread <= '0';
            if(clklast = '0' and clk_reg = '1') then
                io_SPI_mosi <= datareadshiftregister(0);
                io_SPI_D1   <= datareadshiftregister(1);
                io_SPI_D2   <= datareadshiftregister(2);
                io_SPI_D3   <= datareadshiftregister(3);
                datareadshiftregister(27 downto 0) <= datareadshiftregister(31 downto 4);
                nibblecount     <= nibblecount +1;
                haveread        <= '1';
            end if;

            if(nibblecount mod 8 = 0 and nibblecount > 0 and haveread = '1')then
                datareadshiftregister <= data_to_arria;
                next_data         <= '1';
            end if;
			if(nibblecount mod 8 = 1 and haveread = '1') then
                addroffset_int    <= addroffset_int + 1;
            end if;


            if(csn_reg = '1') then
                spistate <= idle;
            end if;

        when others =>
            io_SPI_mosi     <= 'Z';
            io_SPI_miso     <= 'Z';
            io_SPI_D1       <= 'Z';
            io_SPI_D2       <= 'Z';
            io_SPI_D3       <= 'Z';
            spistate        <= idle;
        end case;
    end if;
    end process;

end architecture RTL;
