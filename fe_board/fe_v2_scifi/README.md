#Pinout information for DAB-1 to SciFi board v2

##Pinout on DAB-FMB connector
Same as SciFi - Mutrig1MB connector, should be correct.
No changes / flips seen.

##FEB connector is CON2
Listing as top - bottom, left - right.
A ~ denotes a polarity flip on the connector in the sense of n-above-p

|-------------|--------------|----------|----------|
| FMB PIN     | FEB PIN      | polarity | FPGA pin |
|-------------|--------------|----------|----------|
| SPI_CSn_1_1 | spare_out ~  | inverted | AG16     |
| SPI_CSn_1_2 | clock_aux ~  | inverted | AG6      |
| SPI_CSn_1_3 | fast_reset_B~| inverted | AE11     |
| SPI_CSn_1_4 | clock_B      | normal   | AA10     |
|-------------|--------------|----------|----------|
| RST_1       | fast_reset_A~| inverted | AD22     |
| SCLK_1      | clock_A ~    | inverted | AC21     |
| MISO        | data_in_A_9 ~| inverted | AB17     |
|-------------|--------------|----------|----------|
| INJECTION   | SIN_B ~      | inverted | AB18     |
|-------------|--------------|----------|----------|
| CLK         | SI1_clkA     |          | n.a.     |
| MOSI        | SIN_A ~      | inverted | AK19     |
| DIN_1_1 ~   | data_in_A_1 ~| inv(DAB) | AB22     |
| DIN_1_2 ~   | data_in_A_2 ~| inv(DAB) | AH23     |
| DIN_1_3 ~   | data_in_A_3 ~| inv(DAB) | AG22     |
| DIN_1_4 ~   | data_in_A_4 ~| inv(DAB) | AG21     |
|-------------|--------------|----------|----------|

