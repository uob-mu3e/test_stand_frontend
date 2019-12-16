LIBRARY ieee;
USE ieee.std_logic_1164.ALL;

ENTITY version_reg IS
PORT (
    data_out : OUT STD_LOGIC_VECTOR(27 downto 0)
);
end entity;

ARCHITECTURE rtl OF version_reg IS

BEGIN

    data_out <= X"1be1f41";

end architecture;
