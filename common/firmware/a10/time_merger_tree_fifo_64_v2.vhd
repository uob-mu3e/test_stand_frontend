library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;


entity time_merger_tree_fifo_64_v2 is
generic (
    TREE_w          : positive  := 8;
    TREE_r          : positive  := 8;
    r_width         : integer   := 64;
    w_width         : integer   := 64;
    compare_fifos   : integer   := 32;
    gen_fifos       : integer   := 16--;
);
port (
    -- input
    i_data          : in work.util.slv76_array_t(compare_fifos - 1 downto 0);
    i_rdempty       : in std_logic_vector(compare_fifos - 1 downto 0);
    i_rdreq         : in std_logic_vector(gen_fifos - 1 downto 0);
    i_merge_state   : in std_logic;
    i_mask_n        : in std_logic_vector(compare_fifos - 1 downto 0);

    -- output
    o_q             : out work.util.slv76_array_t(gen_fifos - 1 downto 0);
    o_rdempty       : out std_logic_vector(gen_fifos - 1 downto 0);
    o_rdreq         : out std_logic_vector(compare_fifos - 1 downto 0);
    o_mask_n        : out std_logic_vector(gen_fifos - 1 downto 0);
    
    i_reset_n       : in    std_logic;
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of time_merger_tree_fifo_64_v2 is

    -- merger signals
    constant size : integer := compare_fifos/2;
    
    signal data, data_reg : work.util.slv76_array_t(gen_fifos - 1 downto 0);
    signal layer_state : work.util.slv4_array_t(gen_fifos - 1 downto 0);
    signal wrreq, wrfull, reset_fifo : std_logic_vector(gen_fifos - 1 downto 0);

begin

    tree_fifos:
    FOR i in 0 to gen_fifos - 1 GENERATE

        o_mask_n(i) <= i_mask_n(i) or i_mask_n(i + size);
        reset_fifo(i) <= '0' when i_merge_state = '1' else '1';

        e_link_fifo : entity work.ip_dcfifo_mixed_widths
        generic map(
            ADDR_WIDTH_w    => TREE_w,
            DATA_WIDTH_w    => w_width,
            ADDR_WIDTH_r    => TREE_r,
            DATA_WIDTH_r    => r_width,
            DEVICE          => "Arria 10"--,
        )
        port map (
            aclr    => not i_reset_n or reset_fifo(i),
            data    => data(i),
            rdclk   => i_clk,
            rdreq   => i_rdreq(i),
            wrclk   => i_clk,
            wrreq   => wrreq(i),
            q       => o_q(i),
            rdempty => o_rdempty(i),
            wrfull  => wrfull(i)--,
        );

        wrreq(i) <= '1' when i_merge_state = '1' and wrfull(i) = '0' and i_rdempty(i) = '0' and i_rdempty(i + size) = '0' and and_reduce(layer_state(i)(1 downto 0)) = '0' else '0';
        
        data(i)(37 downto 0) <= i_data(i)(37 downto 0)      when i_mask_n(i) = '1' and i_mask_n(i+size) = '0' else
                                i_data(i+size)(37 downto 0) when i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else
                                data_reg(i)(37 downto 0)    when layer_state(i)(2) = '1' or layer_state(i)(3) = '1' else
                                i_data(i)(75 downto 38)     when layer_state(i)(0) = '1' and i_data(i)(69 downto 66) <= i_data(i+size)(69 downto 66) else
                                i_data(i+size)(75 downto 38)when layer_state(i)(0) = '1' and i_data(i)(69 downto 66) >  i_data(i+size)(69 downto 66) else
                                i_data(i)(75 downto 38)     when layer_state(i)(1) = '1' and i_data(i)(69 downto 66) <= i_data(i+size)(69 downto 66) else
                                i_data(i+size)(75 downto 38)when layer_state(i)(1) = '1' and i_data(i)(69 downto 66) >  i_data(i+size)(69 downto 66) else
                                i_data(i)(37 downto 0)      when i_data(i)(31 downto 28) <= i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66)        <= i_data(i+size)(31 downto 28) else
                                i_data(i+size)(37 downto 0) when i_data(i+size)(31 downto 28) <= i_data(i)(31 downto 28) and i_data(i+size)(69 downto 66)   <= i_data(i)(31 downto 28) else
                                i_data(i)(37 downto 0)      when i_data(i)(31 downto 28) <= i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66)        >  i_data(i+size)(31 downto 28) else
                                i_data(i+size)(37 downto 0) when  i_data(i)(31 downto 28) > i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66)        <= i_data(i+size)(31 downto 28) else
                                (others => '0');
        
        data(i)(75 downto 38)<= i_data(i)(75 downto 38)     when i_mask_n(i) = '1' and i_mask_n(i+size) = '0' else
                                i_data(i+size)(75 downto 38)when i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else
                                i_data(i)(37 downto 0)      when layer_state(i)(2) = '1' and i_data(i)(31 downto 28) <= i_data(i+size)(69 downto 66) else
                                i_data(i+size)(75 downto 38)when layer_state(i)(2) = '1' and i_data(i)(31 downto 28) >  i_data(i+size)(69 downto 66) else
                                i_data(i)(75 downto 38)     when layer_state(i)(3) = '1' and i_data(i+size)(31 downto 28) >  i_data(i)(69 downto 66) else
                                i_data(i+size)(37 downto 0) when layer_state(i)(3) = '1' and i_data(i+size)(31 downto 28) <= i_data(i)(69 downto 66) else
                                (others => '0')             when layer_state(i)(0) = '1' and i_data(i)(69 downto 66) <= i_data(i+size)(69 downto 66) else
                                (others => '0')             when layer_state(i)(0) = '1' and i_data(i)(69 downto 66) >  i_data(i+size)(69 downto 66) else
                                (others => '0')             when layer_state(i)(1) = '1' and i_data(i)(69 downto 66) <= i_data(i+size)(69 downto 66) else
                                (others => '0')             when layer_state(i)(1) = '1' and i_data(i)(69 downto 66) >  i_data(i+size)(69 downto 66) else
                                i_data(i)(75 downto 38)     when i_data(i)(31 downto 28) <= i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66)        <= i_data(i+size)(31 downto 28) else
                                i_data(i+size)(75 downto 38)when i_data(i+size)(31 downto 28) <= i_data(i)(31 downto 28) and i_data(i+size)(69 downto 66)   <= i_data(i)(31 downto 28) else
                                i_data(i+size)(37 downto 0) when i_data(i)(31 downto 28) <= i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66)        >  i_data(i+size)(31 downto 28) else
                                i_data(i)(37 downto 0)      when i_data(i)(31 downto 28) > i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66)         <= i_data(i+size)(31 downto 28) else
                                (others => '0');

        o_rdreq(i) <= '1' when i_mask_n(i) = '1' and i_mask_n(i+size) = '0' else
                      '1' when i_data(i)(31 downto 28) <= i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66) <= i_data(i+size)(31 downto 28) else
                      '0';

        o_rdreq(i+size) <=  '1' when i_mask_n(i+size) = '1' and i_mask_n(i) = '0' else
                            '1' when i_data(i+size)(31 downto 28) <= i_data(i)(31 downto 28) and i_data(i+size)(69 downto 66) <= i_data(i)(31 downto 28) else
                            '0';
        
        process(i_clk, i_reset_n)
        begin
        if ( i_reset_n /= '1' ) then
            layer_state(i) <= (others => '0');
        elsif ( rising_edge(i_clk) ) then
            if ( layer_state(i)(0) = '1' or layer_state(i)(1) <= '1' ) then
                data_reg(i) <= data(i);
                if ( i_data(i)(69 downto 66) <= i_data(i+size)(69 downto 66) ) then
                    layer_state(i)(2) <= '1';
                else
                    layer_state(i)(2) <= '0';
                end if;
                if ( i_data(i)(69 downto 66) > i_data(i+size)(69 downto 66) ) then
                    layer_state(i)(3) <= '1';
                else
                    layer_state(i)(3) <= '0';
                end if;
                layer_state(i)(0) <= '0';
                layer_state(i)(1) <= '0';
            else
                if ( i_data(i)(31 downto 28) <= i_data(i+size)(31 downto 28) and i_data(i)(69 downto 66) > i_data(i+size)(31 downto 28) ) then
                    layer_state(i)(0) <= '1';
                else
                    layer_state(i)(0) <= '0';
                end if;
                
                if ( i_data(i)(31 downto 28) > i_data(i+size)(31 downto 28) and i_data(i)(31 downto 28) <= i_data(i+size)(69 downto 66) ) then
                    layer_state(i)(1) <= '1';
                else
                    layer_state(i)(1) <= '0';
                end if;
            end if;
        end if;
        end process;

    END GENERATE;

end architecture;
