#!/usr/bin/env python3
# TMC drivers registers calibration tool (plotter)
#
# Copyright (C) 2024  Alexander Fedorov <altzbox@gmail.com>
# Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license.

#################################################################################################################
RESULTS_FOLDER = "~/printer_data/config/adxl_results/chopper_magnitude"
DATA_FOLDER = "/tmp/"
#################################################################################################################

import os, sys, csv
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime

RESULTS_FOLDER = os.path.expanduser(RESULTS_FOLDER)
CUTOFF_RANGE = 5


def cleaner():
    os.system("rm -f /tmp/*.csv")
    sys.exit(0)


def check_export_path(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            print(f"Error generate path {path}: {e}")


def parse_arguments():
    args = sys.argv[1:]
    parsed_args = {"iterations": 1}
    for arg in args:
        if "=" in arg:
            name, value = arg.split("=")
            parsed_args[name] = int(value) if value.isdigit() else value
    return parsed_args


def calc_static_magnitude(file):
    data = np.array(
        [
            [float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])]
            for row in csv.DictReader(file)
        ]
    )
    return np.mean(data, axis=0)


def calc_magnitude(file, static_data):
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
    md_magnitude = np.median(np.linalg.norm(data, axis=1))
    return md_magnitude


def main():
    print("Magnitude graphs generation...")
    args = parse_arguments()
    iterations = int(args.get("iterations", 1))
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Calc static magnitude
    static_name = next(
        (name for name in os.listdir(DATA_FOLDER) if name.endswith("stand_still.csv")),
        None,
    )
    if static_name is None:
        print("No stand_still.csv found in /tmp; aborting plot generation")
        return
    with open(f"{DATA_FOLDER}{static_name}", "r") as file:
        static_data = calc_static_magnitude(file)
        accel_chip = static_name.split("-")[0]
    # Calc magnitudes on parameters
    samples = {}
    datapoint = []
    empty_error = 0
    data_files = sorted(
        os.listdir(DATA_FOLDER),
        key=lambda x: os.path.getmtime(os.path.join(DATA_FOLDER, x)),
        reverse=True,
    )
    for name in data_files:
        if name.endswith("__.csv"):
            parts = name.split("__")[1].split("_")
            params_map = {}
            for part in parts:
                for key in ["eh", "tbl", "toff", "speed", "iter"]:
                    if part.startswith(key):
                        params_map[key] = part[len(key) :]
                        break
            if not all(k in params_map for k in ["eh", "tbl", "toff", "speed", "iter"]):
                continue
            extra_hyst = int(params_map["eh"])
            tbl = int(params_map["tbl"])
            toff = int(params_map["toff"])
            speed_val = float(params_map["speed"]) / 100.0
            iter_val = int(params_map["iter"])
            out_name = (
                f"extra_hyst={extra_hyst}_tbl={tbl}_toff={toff}_speed={speed_val:.2f}"
            )
            with open(f"{DATA_FOLDER}{name}", "r") as file:
                try:
                    md_magnitude = calc_magnitude(file, static_data)
                    datapoint.append(md_magnitude)
                    if iter_val == iterations:
                        samples[out_name] = np.mean(datapoint, axis=0)
                        datapoint.clear()
                except Exception:
                    datapoint.clear()
                    empty_error += 1
                    samples[out_name] = 0

    if not samples:
        print("No measurement CSV files found to plot.")
        return

    # Graphs generation
    colors = [
        "#808080",  # Gray for toff=0 (auto-calc)
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
        list(reversed(list(samples.items()))),
        sorted(samples.items(), key=lambda x: x[1]),
    ]
    names = ["", "sorted_"]
    plot_html_path = None
    for param, name in zip(params, names):
        fig = go.Figure()
        for entry in param:
            toff_val = int(entry[0].split("_")[2].split("=")[1])
            # Use modulo to wrap colors for toff values beyond the palette
            color = colors[toff_val % len(colors)]
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
            f"{name}autotune_sweep_{accel_chip}_{now}.html",
        )
        pio.write_html(fig, plot_html_path, auto_open=False)

    # Export Info
    if plot_html_path:
        print(f"Access to interactive plot at: {plot_html_path}")
    if empty_error:
        print(
            f"Warning!!! Empty data cells detected ({empty_error}), make sure you dont run out of memory"
        )


if __name__ == "__main__":
    if sys.argv[1] == "cleaner":
        cleaner()
    check_export_path(RESULTS_FOLDER)
    main()
