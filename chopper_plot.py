#!/usr/bin/env python3
"""TMC drivers registers calibration tool (plotter).

Copyright (C) 2024  Alexander Fedorov <altzbox@gmail.com>
Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>

This file may be distributed under the terms of the GNU GPLv3 license.
"""

# Standard Library Imports
import csv
import glob
import os
import sys
from datetime import datetime
from typing import TextIO

import numpy as np

# Third Party Imports
import plotly.graph_objects as go
import plotly.io as pio

RESULTS_FOLDER = os.path.expanduser(
    "~/printer_data/config/adxl_results/chopper_magnitude"
)
DATA_FOLDER = os.path.expanduser(
    "~/printer_data/config/adxl_results/chopper_magnitude/tmp"
)

FCLK = 12  # MHz
CUTOFF_RANGE = 5


def cleaner() -> None:
    """Clean temporary data files and exit."""
    for f in glob.glob(os.path.join(DATA_FOLDER, "*.csv")):
        os.remove(f)
    sys.exit(0)


def check_export_path(path: str) -> None:
    """Check and create export path if it doesn't exist.

    Args:
        path (str): The directory path to check/create.
    """
    if os.path.exists(path):
        return
    try:
        os.makedirs(path)
    except OSError as e:
        print(f"Error generate path {path}: {e}")


def parse_arguments() -> dict:
    """Parse command line arguments.

    Returns:
        dict: Parsed arguments as a dictionary.
    """
    args = sys.argv[1:]
    parsed_args = {}
    for arg in args:
        name, value = arg.split("=")
        parsed_args[name] = int(value) if value.isdigit() else value
    return parsed_args


def calc_static_magnitude(file: TextIO) -> np.ndarray:
    """Calculate static acceleration data from CSV file.

    Args:
        file (TextIO): Opened CSV file containing static acceleration data.

    Returns:
        np.ndarray: Mean static acceleration values for x, y, z axes.
    """
    data = np.array(
        [
            [float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])]
            for row in csv.DictReader(file)
        ]
    )
    return np.mean(data, axis=0)


def calc_magnitude(file: TextIO, static_data: np.ndarray) -> float:
    """Calculate median magnitude of acceleration data from CSV file.

    Args:
        file (TextIO): Opened CSV file containing acceleration data.
        static_data (np.ndarray): Mean static acceleration values for x, y, z
            axes.

    Returns:
        float: Median magnitude of acceleration data.
    """
    data = (
        np.array(
            [
                [float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])]
                for row in csv.DictReader(file)
            ]
        )
        - static_data
    )
    trim_size = len(data) // CUTOFF_RANGE
    data = data[trim_size:-trim_size]
    return np.median(np.linalg.norm(data, axis=1))


def main() -> None:
    """Main function to process data files and generate magnitude graphs."""
    print("Magnitude graphs generation...")
    args = parse_arguments()
    driver = args.get("driver")
    iterations = args.get("iterations")
    sense_resistor = round(float(args.get("sense_resistor")), 3)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Calc static magnitude
    static_name = next(
        (name for name in os.listdir(DATA_FOLDER) if name.endswith("stand_still.csv")),
        None,
    )
    with open(os.path.join(DATA_FOLDER, static_name)) as file:
        static_data = calc_static_magnitude(file)
        accel_chip = static_name.split("-")[0]
    # Calc magnitudes on registers
    samples = {}
    datapoint = []
    empty_error = 0
    data_files = sorted(
        os.listdir(DATA_FOLDER),
        key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)),
        reverse=True,
    )
    for name in data_files:
        if not name.endswith("__.csv"):
            continue

        with open(os.path.join(DATA_FOLDER, name)) as file:
            curr, tbl, toff, hstrt, hend, tpfd, speed, freq, iteration = name.split(
                "__"
            )[1].split("_")
            out_name = (
                f"current={curr}_"
                f"tbl={tbl}_"
                f"toff={toff}_"
                f"hstrt={hstrt}_"
                f"hend={hend}_"
                f"tpfd={tpfd}_"
                f"speed={float(speed) / 100:.2f}_"
                f"freq={float(freq) / 1000:.2f}kHz"
            )
            try:
                md_magnitude = calc_magnitude(file, static_data)
                datapoint.append(md_magnitude)
                if int(iteration) == iterations:
                    samples[out_name] = np.mean(datapoint, axis=0)
                    datapoint.clear()
            except Exception:
                datapoint.clear()
                empty_error += 1
                samples[out_name] = 0

    # Graphs generation
    colors = [
        "",
        "#2F4F4F",
        "#12B57F",
        "#9DB512",
        "#DF8816",
        "#1297B5",
        "#5912B5",
        "#B51284",
        "#127D0C",
    ]
    params = [
        reversed(list(samples.items())),
        sorted(samples.items(), key=lambda x: x[1]),
    ]
    names = ["", "sorted_"]
    for param, name in zip(params, names):
        fig = go.Figure()
        for entry in param:
            toff = int(entry[0].split("_")[2].split("=")[1])
            color = colors[toff if toff <= 8 else toff - 8]
            fig.add_trace(
                go.Bar(
                    x=[entry[1]],
                    y=[entry[0]],
                    marker_color=color,
                    orientation="h",
                    showlegend=False,
                )
            )
        fig.update_layout(
            title="Median Magnitude vs Parameters",
            xaxis_title="Median Magnitude",
            yaxis_title="Parameters",
            coloraxis_showscale=True,
        )
        plot_html_path = os.path.join(
            RESULTS_FOLDER,
            f"{name}interactive_plot"
            f"_{accel_chip}_"
            f"tmc{driver}_"
            f"{sense_resistor}_"
            f"{now}.html",
        )
        pio.write_html(fig, plot_html_path, auto_open=False)
        speed1 = params[1][0][0].split("_")[6].split("=")[1]
        speed2 = params[1][1][0].split("_")[6].split("=")[1]
        if speed1 != speed2:
            break

    # Export Info
    if names[1] in plot_html_path:
        plot_html_path = "/".join(
            [*plot_html_path.split("/")[:-1], plot_html_path.split(names[1])[1]]
        )

    print(f"Access to interactive plot at: {plot_html_path}")

    if empty_error:
        print(
            f"Warning!!! Empty data cells detected ({empty_error}), "
            "make sure you dont run out of memory"
        )


if __name__ == "__main__":
    if sys.argv[1] == "cleaner":
        cleaner()
    check_export_path(RESULTS_FOLDER)
    main()
