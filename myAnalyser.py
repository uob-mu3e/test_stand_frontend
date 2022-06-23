#!/usr/bin/python3
"""
Analyse environmental variables collected through MIDAS
"""

import sys

sys.path.append("midas/python/")
import midas.file_reader
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    FILE I/O and driver
    """
    try:
        mfile = midas.file_reader.MidasFile(f"online/run{sys.argv[1].zfill(5)}.mid")
    except IndexError:
        print("Please provide a run number as an argument.")
    read_plot_file(mfile)


def read_plot_file(mfile):
    """
    Parse file and produce and save plots
    """
    # DATAFRAME INITIALIZATION
    data = {
        "Temp": [],
        "Flow": [],
        "PWM": [],
        "Setpoint": [],
        "Relative Humidity": [],
        "Ambient Temperature": [],
    }
    while mfile.read_next_event_header():
        header = mfile.event.header
        if header.is_midas_internal_event():
            # Skip over events that contain midas messages or ODB dumps
            continue

        mfile.read_this_event_body()
        data_tmp = []
        for name, bank in mfile.event.banks.items():
            data_tmp.append(np.array(bank.data))
        data["Temp"].append(data_tmp[0])
        data["Flow"].append(data_tmp[1])
        data["PWM"].append(data_tmp[2])

        """
        Variable _A_, not necessary for A&J's project but might be useful to add back in?
        TODO: put this back in the original Arduino firmware. FE.cpp seems to contain it already.
        """
        # data["Average Airflow"].append(data_tmp[3])

        # data["Setpoint"].append(data_tmp[4])  # commented out in FE.cpp so not relevant anymore
        data["Relative Humidity"].append(data_tmp[4])
        data["Ambient Temperature"].append(data_tmp[5])

        plt.clf()
        plt.tight_layout()
        plt.scatter(
            range(1, len(data["Temp"]) + 1),
            data["Temp"],
            s=10,
            c="steelblue",
            marker="x",
            label="Temperature",
        )
        plt.scatter(
            range(1, len(data["Flow"]) + 1),
            data["Flow"],
            s=10,
            c="forestgreen",
            marker="x",
            label="Flow",
        )
        plt.scatter(
            range(1, len(data["PWM"]) + 1),
            data["PWM"],
            s=10,
            c="orangered",
            marker="x",
            label="PWM",
        )
        plt.scatter(
            range(1, len(data["Setpoint"]) + 1),
            data["Setpoint"],
            s=10,
            c="pink",
            marker="x",
            label="Setpoint",
        )
        plt.scatter(
            range(1, len(data["Relative Humidity"]) + 1),
            data["Relative Humidity"],
            s=10,
            c="black",
            marker="x",
            label="Relative Humidity",
        )
        plt.scatter(
            range(1, len(data["Ambient Temperature"]) + 1),
            data["Ambient Temperature"],
            s=10,
            c="cyan",
            marker="x",
            label="Ambient Temperature",
        )
        plt.legend(loc=(0.75, 0.75))
        plt.xlabel("time (s)")
        plt.xlim()
        plt.ylim(-5, 50)
        plt.savefig(f"online/run{sys.argv[1]}.png", bbox_inches="tight")

    # print("Overall size of event,type ID and bytes")
    # print((header.serial_number, header.event_id, header.event_data_size_bytes))
    # if isinstance(bank.data, tuple) and len(bank.data):
    #     # A tuple of ints/floats/etc (a tuple is like a fixed-length list)
    #     type_str = "tuple of %s containing %d elements" % (type(bank.data[0]).__name__, len(bank.data))
    # elif isinstance(bank.data, tuple):
    #     # A tuple of length zero
    #     type_str = "empty tuple"
    # elif isinstance(bank.data, str):
    #     # Of the original data was a list of chars, we convert to a string.
    #     type_str = "string of length %d" % len(bank.data)
    # else:
    #     # Some data types we just leave as a set of bytes.
    #     type_str = type(bank.data[0]).__name__

    # print("  - bank %s contains %d bytes of data. Python data type: %s" % (name, bank.size_bytes, type_str))


if __name__ == "__main__":
    main()
