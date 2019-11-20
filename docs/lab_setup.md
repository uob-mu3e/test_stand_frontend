# lab setup (2019-06-19)

```
    +--------------------------------------------------+
    |            General Frontend settings             |
    |             System active:             |   [x]   |
    |                 Delay:                 |    0    |
    |   [Reset SC Master] [Reset SC Slave]   |         |
    +--------------------------------------------------+

    +--------------------------------------------------+
    |                     Read SC                      |
    |                   [Read]                   |     |
    | FPGA_ID                                    | 0   |
    | Start Add.                                 | 0   |
    | Length                                     | 0   |
    | PCIe MEM Start                             | 0   |
    +--------------------------------------------------+

    +--------------------------------------------------+
    |                     Write SC                     |
    | [Write] []                                   |   |
    | FPGA_ID                                      | 0 |
    | Start Add.                                   | 0 |
    | PCIe MEM Start                               | 0 |
    |                [Single Write]                |   |
    | FPGA_ID                                      | 0 |
    | Start Add.                                   | 0 |
    | Data                                         | 0 |
    | PCIe MEM Start                               | 0 |
    +--------------------------------------------------+

    +--------------------------------------------------+
    |                 Read Memory PCIe                 |
    |                                 | Start | Length |
    | [Load RM Mem] [Read Memory]     | 0     | 0      |
    +--------------------------------------------------+

    +--------------------------------------------------+
    |                Write Memory PCIe                 |
    |                                 | Start | Length |
    | [Load RM Mem] [Write Memory]    | 0     | 0      |
    +--------------------------------------------------+
```

## `[Reset SC Master]`

This part of the firmware checks if there is a SC command in the PCIe Write Memory.
It will look at address `0` for the start word `0xBAD` and will then loop until the stop word `0x9C`.
Then it will wait at address `0 + length` until `0x9C`.
The reset sets it back to address `0`.

## `[Reset SC Slave]`

This part of the firmware checks if there is a SC command on the link
and it will write it to the PCIe Read Memory.
At the moment this feature is not fully running.

## `Read SC / [Read]`

With this one can send a read SC command to the FEB.

- `FPGA_ID`: 0 means send via all channels; at the moment let it at 0.
- `Start Add.`: set start address for the ram on the FEB
- `Length`: set length for ram on FEB
- `PCIe MEM Start`: address for the PCIe Write Memory on the Arria10

## `Write SC / [Write]`

With this one can send a write SC command to the FEB for multi address.
First upload a `.csv` file with the data in decimal for each address starting from the `Start Add.`
An example is in switching_pc/midas_fe/test_sc.csv

- `FPGA_ID`: 0 means send via all channels; at the moment let it at 0.
- `Start Add.`: set the start address for the ram on the FEB
- `PCIe MEM Start`: address for the PCIe Write Memory on the Arria10

## `Write SC / [Signle Write]`

With this one can send a single write sc command to the FEB.

- `FPGA_ID`: 0 means send via all channels; at the moment let it at 0.
- `Start Add.`: set address for the ram on the FEB
- `Data`: data in decimal
- `PCIe MEM Start`: address for the PCIe Write Memory on the Arria10

## `Read Memory PCIe` / `Write Memory PCIe`

One can read the PCIe Read and Write Memory of the Arria10 board with this section.
First specify the start address and length then click read memory and the load mem.
