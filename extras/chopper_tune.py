"""Chopper Tune extension for Klipper.

TMC drivers registers calibration tool.

Copyright (C) 2024  Alexander Fedorov <altzbox@gmail.com>
Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>

This file may be distributed under the terms of the GNU GPLv3 license.
"""

# Standard Library Imports
from __future__ import annotations

import csv
import glob
import multiprocessing
import os
import re
import shutil
import time
import traceback
from enum import IntEnum
from functools import cache, reduce, wraps
from typing import TYPE_CHECKING, Callable

# Third-Party Imports
import numpy as np
from scipy.optimize import differential_evolution

# Klipper Imports

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    from configfile import ConfigWrapper
    from gcode import GCodeCommand, GCodeDispatch
    from klippy import Printer
    from reactor import PollReactor
    from toolhead import ToolHead

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

IS_DIGIT = re.compile(r"[0-9\-.]+")

DEFAULT_ACCEL_CHIP = "adxl345"
RESULTS_FOLDER = os.path.expanduser(
    "~/printer_data/config/adxl_results/chopper_magnitude"
)
DATA_FOLDER = os.path.expanduser(
    "~/printer_data/config/adxl_results/chopper_magnitude/tmp"
)

FCLK = 12  # MHz
CUTOFF_RANGE = 5


def gcmd_grabber(f: Callable) -> Callable:
    """Decorator to grab the gcmd arg temporarily from command methods.

    This allows non-command methods to use the respond_info and respond_debug
    methods.

    Args:
        f (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function.
    """

    @wraps(f)
    def wrapped_f(self: ChopperTune, gcmd: GCodeCommand, *args, **kwargs) -> None:
        self._gcmd = gcmd
        result = f(self, gcmd, *args, **kwargs)
        self._gcmd = None
        return result

    return wrapped_f


class MeasurementMode(IntEnum):
    """Integer enumerator to specify the current measurement mode."""

    Resonances = 0
    Vibrations = 1

    def __repr__(self) -> str:
        """Return the enum name for str().

        Returns:
            str: The name as the string representation of this MeasurementMode.
        """
        return self.name

    __str__ = __repr__

    @classmethod
    def to_mode(cls, mode: int | str | MeasurementMode) -> MeasurementMode:
        """Convert the given mode value to a MeasurementMode enum.

        Args:
            mode (int | str | MeasurementMode]): The value to convert to a
                MeasurementMode.

        Raises:
            TypeError: Input value type is invalid.
            ValueError: Input value is invalid.

        Returns:
            MeasurementMode: The enum.
        """
        if not isinstance(mode, (int, str, MeasurementMode)):
            raise TypeError(
                "mode should be a MeasurementMode enum value or one of "
                f"{[m.name for m in cls] + [m.value for m in cls]}, "
                f"not {mode.__class__.__name__}: '{mode}'"
            )
        if isinstance(mode, str):
            mode_name_lut = {m.name.lower(): m.name for m in cls}
            mode_name_lut.update({m.value: m.name for m in cls})
            mode_lower_case = mode.lower()
            if mode_lower_case not in mode_name_lut:
                raise ValueError(
                    "mode should be a MeasurementMode enum value or one of "
                    f"{[m.name for m in cls] + [m.value for m in cls]}, "
                    f"not '{mode}'"
                )

            return cls.__members__[mode_name_lut[mode_lower_case]]

        return mode


class SearchMethod(IntEnum):
    """Integer enumerator to specify the current search method."""

    BruteForce = 0
    Adaptive = 1

    def __repr__(self) -> str:
        """Return the enum name for str().

        Returns:
            str: The name as the string representation of this SearchMethod.
        """
        return self.name

    __str__ = __repr__

    @classmethod
    def to_method(cls, method: int | str | SearchMethod) -> SearchMethod:
        """Convert the given method value to a SearchMethod enum.

        Args:
            method (int | str | SearchMethod]): The value to convert to a
                SearchMethod.

        Raises:
            TypeError: Input value type is invalid.
            ValueError: Input value is invalid.

        Returns:
            SearchMethod: The enum.
        """
        if not isinstance(method, (int, str, SearchMethod)):
            raise TypeError(
                "method should be a SearchMethod enum value or one of "
                f"{[m.name for m in cls] + [m.value for m in cls]}, "
                f"not {method.__class__.__name__}: '{method}'"
            )
        if isinstance(method, str):
            method_name_lut = {m.name.lower(): m.name for m in cls}
            method_name_lut.update({m.value: m.name for m in cls})
            method_lower_case = method.lower()
            if method_lower_case.replace("_", "") not in method_name_lut:
                raise ValueError(
                    "method should be a SearchMethod enum value or one of "
                    f"{[m.name for m in cls] + [m.value for m in cls]}, "
                    f"not '{method}'"
                )

            return cls.__members__[method_name_lut[method_lower_case.replace("_", "")]]

        return method


class AccelerometerMeasure:
    """A context manager that helps with accelerometer measurement."""

    def __init__(
        self,
        printer: Printer,
        gcode: GCodeDispatch,
        accel_chip: str,
        name: str,
    ) -> None:
        self.printer = printer
        self.gcode = gcode
        self.accel_chip = accel_chip
        self.name = name

    @property
    def full_name(self) -> str:
        """Return the full name of the measurement file.

        Returns:
            str: The full name of the measurement file.
        """
        return f"{self.accel_chip}-{self.name}.csv"

    @property
    def full_path(self) -> str:
        """Return the full path of the measurement file.

        Returns:
            str: The full path of the measurement file.
        """
        # Klipper saves the measurement files in /tmp/
        return f"/tmp/{self.full_name}"  # noqa: S108

    def __enter__(self) -> Self:
        """Enter to the context."""
        self.gcode.run_script_from_command(
            f"ACCELEROMETER_MEASURE CHIP={self.accel_chip} NAME={self.name}"
        )
        return self

    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        tb: None | TracebackType,
    ) -> None:
        """Exit the context.

        Ignore the exceptions, if any, Klipper will handle it.
        """
        self.gcode.run_script_from_command(
            f"ACCELEROMETER_MEASURE CHIP={self.accel_chip} NAME={self.name}"
        )

    def move(self) -> str:
        """Move the measurement file over the DATA_FOLDER.

        Returns:
            str: The final destination path of the measurement file.
        """
        if not os.path.exists(DATA_FOLDER):
            self.gcode.respond_info(f"Data folder doesn't exist: {DATA_FOLDER}")
            self.gcode.respond_info(f"Creating: {DATA_FOLDER}")
            os.makedirs(DATA_FOLDER, exist_ok=True)
        destination = os.path.join(DATA_FOLDER, self.full_name)
        if os.path.exists(destination):
            # remove the previous file
            os.remove(destination)

        def do_threaded_move() -> None:
            """Move the measurement file in another thread.

            ...so the main process is not blocked.
            """
            start_time = time.time()
            max_wait_time = 10  # seconds
            prev_size = -1
            curr_size = (
                0
                if not os.path.exists(self.full_path)
                else os.path.getsize(self.full_path)
            )
            while not os.path.exists(self.full_path) or prev_size != curr_size:
                time.sleep(0.1)
                if os.path.exists(self.full_path):
                    prev_size = curr_size
                    curr_size = os.path.getsize(self.full_path)
                if (time.time() - start_time) > max_wait_time:
                    break

            if os.path.exists(self.full_path):
                shutil.move(self.full_path, destination)
            else:
                self.gcode.respond_info(f"File doesn't exist: {self.full_path}")

        move_proc = multiprocessing.Process(target=do_threaded_move)
        move_proc.daemon = True
        move_proc.start()

        return destination

    def get_full_path(self) -> str:
        """Get the data full path.

        Before returning the data path, ensure the file has been written.

        Returns:
            str: The final destination path of the measurement file.
        """
        start_time = time.time()
        max_wait_time = 10  # seconds
        prev_size = -1
        curr_size = (
            0 if not os.path.exists(self.full_path) else os.path.getsize(self.full_path)
        )
        while not os.path.exists(self.full_path) or prev_size != curr_size:
            time.sleep(0.1)
            if os.path.exists(self.full_path):
                prev_size = curr_size
                curr_size = os.path.getsize(self.full_path)
            if (time.time() - start_time) > max_wait_time:
                break

        if not os.path.exists(self.full_path):
            self.gcode.respond_info(f"File doesn't exist: {self.full_path}")

        return self.full_path


class Coord(list):
    """Custom "list" class for coordinates - add easy access to x, y, z components.

    The difference between the gcode.Coord is that, this class allows attribute
    setting.

    Args:
        t (Coord | list | tuple): Another Coord instance or a list or a tuple.
    """

    __slots__ = ()

    def __new__(cls, t: Coord | list | tuple) -> Self:
        """Create a new Coord instance."""
        if len(t) < 4:
            t = list(tuple(t) + (0,) * (3 - len(t)))
        return list.__new__(cls, t)

    @property
    def x(self) -> float:
        """Return the x component.

        Returns:
            float: The x component.
        """
        return self[0]

    @x.setter
    def x(self, x: float) -> None:
        """Set the x component.

        Args:
            x (float): The x component value.
        """
        self[0] = x

    @property
    def y(self) -> float:
        """Return the y component.

        Returns:
            float: The y component.
        """
        return self[1]

    @y.setter
    def y(self, y: float) -> None:
        """Set the y component.

        Args:
            y (float): The y component value.
        """
        self[1] = y

    @property
    def z(self) -> float:
        """Return the z component.

        Returns:
            float: The z component.
        """
        return self[2]

    @z.setter
    def z(self, z: float) -> None:
        """Set the z component.

        Args:
            z (float): The z component value.
        """
        self[2] = z

    def length(self) -> float:
        """Return the vector length.

        Returns:
            float: The vector length.
        """
        return float(reduce(lambda x, y: x + y**2, [0, *self]) ** 0.5)

    def unitize(self) -> Self:
        """Make self unit vector."""
        other = self / self.length()
        for i in range(len(self)):
            self[i] = other[i]
        return self

    def __add__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the + operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the addition.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x + other[0], self.y + other[1], self.z + other[2]))
        if isinstance(other, (int, float)):
            return Coord([i + other for i in self])
        return super().__add__(other)

    def __iadd__(self, other: Coord | list | tuple | float) -> Self:
        """Overload the += operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Self: The result of the addition.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x + other[0], self.y + other[1], self.z + other[2]))
        if isinstance(other, (int, float)):
            return Coord([i + other for i in self])
        return super().__iadd__(other)

    def __sub__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the - operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the addition.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x - other[0], self.y - other[1], self.z - other[2]))
        if isinstance(other, (int, float)):
            return Coord([i - other for i in self])
        return super().__sub__(other)

    def __mul__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the * operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the multiplication.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x * other[0], self.y * other[1], self.z * other[2]))
        if isinstance(other, (int, float)):
            return Coord([i * other for i in self])
        return super().__mul__(other)

    def __imul__(self, other: Coord | list | tuple | float) -> Self:
        """Overload the *= operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Self: The result of the multiplication.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x * other[0], self.y * other[1], self.z * other[2]))
        if isinstance(other, (int, float)):
            return Coord([i * other for i in self])
        return super().__imul__(other)

    def __truediv__(self, other: Coord | list | tuple | float) -> Coord:
        """Overload the / operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the division.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x / other[0], self.y / other[1], self.z / other[2]))
        if isinstance(other, (int, float)):
            return Coord([i / other for i in self])
        return super().__truediv__(other)

    def __itruediv__(self, other: Coord | list | tuple | float) -> Self:
        """Overload the /= operator.

        Args:
            other (Coord | list | tuple | float): The other operand.

        Returns:
            Coord: The result of the division.
        """
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x / other[0], self.y / other[1], self.z / other[2]))
        if isinstance(other, (int, float)):
            return Coord([i / other for i in self])
        return super().__itruediv__(other)


class CoordGenerator:
    """A class to generate coordinates/positions for chopper tuning.

    Args:
        direction (Coord | list | tuple): The initial direction.
        start_coord (None | Coord): The starting coordinate.
    """

    def __init__(
        self, direction: Coord | list | tuple, start_coord: None | Coord = None
    ) -> None:
        self._direction = None
        self.direction = direction
        if start_coord is None:
            start_coord = Coord((0, 0, 0))
        self.current_coord = start_coord

    @property
    def direction(self) -> Coord:
        """Return the direction of this CoordGenerator.

        Returns:
            Coord: The direction of this CoordGenerator.
        """
        return self._direction

    @direction.setter
    def direction(self, direction: Coord | list | float) -> None:
        """Set the direction attribute value.

        Args:
            direction (Coord | list | float): The direction value.

        Raises:
            TypeError: If the direction is not a list, tuple or Coord instance.
        """
        if not isinstance(direction, (list, tuple, Coord)):
            raise TypeError(
                f"direction should be a list, tuple or a Coord instance, not "
                f"{direction.__class__.__name__}: '{direction}'"
            )

        if isinstance(direction, (list, tuple)):
            direction = Coord(direction)

        if direction.length() == 0:
            raise ValueError("direction length cannot be zero.")

        self._direction = direction.unitize()

    def switch_direction(self) -> None:
        """Switch direction."""
        self.direction *= (-1, -1, -1)

    def next(self, travel_distance: float) -> float:
        """Get the next position.

        Args:
            travel_distance (float): The travel distance.

        Returns:
            float: The next position.
        """
        self.current_coord += self.direction * travel_distance
        return self.current_coord


def calc_static_magnitude(data_path: str) -> np.ndarray:
    """Calculate static acceleration data from CSV file.

    Args:
        data_path (str): The path to the CSV file containing static
            acceleration data.

    Returns:
        np.ndarray: Mean static acceleration values for x, y, z axes.
    """
    start_time = time.time()
    max_wait_time = 10  # seconds
    prev_size = -1
    curr_size = 0 if not os.path.exists(data_path) else os.path.getsize(data_path)
    while not os.path.exists(data_path) or prev_size != curr_size:
        time.sleep(0.1)  # sleep while the file is getting written
        if os.path.exists(data_path):
            prev_size = curr_size
            curr_size = os.path.getsize(data_path)
        if (time.time() - start_time) > max_wait_time:
            break

    with open(data_path) as file:
        data = np.array(
            [
                [float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])]
                for row in csv.DictReader(file)
            ]
        )
    return np.mean(data, axis=0)


def calc_magnitude(data_path: str, static_data: np.ndarray) -> float:
    """Calculate median magnitude of acceleration data from CSV file.

    Args:
        data_path (str): The path to the CSV file containing acceleration data.
        static_data (np.ndarray): Mean static acceleration values for x, y, z
            axes.

    Returns:
        float: Median magnitude of acceleration data.
    """
    start_time = time.time()
    max_wait_time = 10  # seconds
    prev_size = -1
    curr_size = 0 if not os.path.exists(data_path) else os.path.getsize(data_path)
    while not os.path.exists(data_path) or prev_size != curr_size:
        time.sleep(0.1)  # sleep while the file is getting written
        if os.path.exists(data_path):
            prev_size = curr_size
            curr_size = os.path.getsize(data_path)
        if (time.time() - start_time) > max_wait_time:
            break

    with open(data_path) as file:
        data = (
            np.array(
                [
                    [
                        float(row["accel_x"]),
                        float(row["accel_y"]),
                        float(row["accel_z"]),
                    ]
                    for row in csv.DictReader(file)
                ]
            )
            - static_data
        )
    trim_size = len(data) // CUTOFF_RANGE
    data = data[trim_size:-trim_size]
    return np.median(np.linalg.norm(data, axis=1))


class ChopperTune:
    """The main class to handle the chopper tune functionality.

    Args:
        config (ConfigWrapper): The configuration wrapper.
    """

    def __init__(self, config: ConfigWrapper) -> None:
        self.printer: Printer = config.get_printer()
        self.gcode: GCodeDispatch = self.printer.lookup_object("gcode")
        self.configfile = self.printer.lookup_object("configfile")
        self._settings = None
        self.reactor: PollReactor = self.printer.get_reactor()
        self._driver_settings = {}
        self._stepper_settings = {}
        self.registers = {
            "stepper_count": 0,
            "tbl": -1,
            "toff": -1,
            "hend": -1,
            "hstrt": -1,
            "tpfd": -1,
            "curr": -1,
        }

        # state variables
        self._gcmd = None
        self._toolhead = None

        # config values
        self.debug = config.getboolean("debug", False)
        self.inset = config.getfloat("inset", 10)
        self.current_change_step = config.getint("current_change_step", 25)
        self.measure_time = config.getint("measure_time", 1250)
        self.required_rpm = [
            float(f.strip())
            for f in config.getlist("required_rpm", [str(v) for v in [37.5, 150, 1.5]])
        ]
        self.delay = config.getfloat("delay", 500)
        self.fclk = config.getint("fclk", 12)

        self.kinematics = config.getsection("printer").get("kinematics")

        # runtime variables
        self.driver = None
        self.resistor = None
        self.number_of_samples = 0

        # Calculated values
        self.search_method = None
        self.measurement_mode = MeasurementMode.Vibrations
        self.max_speed = None
        self.travel_speed = None
        self.travel_distance = None
        self.coord_generator = None
        self.accel_chip = None
        self.steppers = None
        self.current = None
        self.static_noise_magnitude = None
        self.initial_position = None
        self.initial_direction = None

        # Bounds
        self.bounds = []
        self.current_min = None
        self.current_max = None
        self.tbl_min = None
        self.tbl_max = None
        self.toff_min = None
        self.toff_max = None
        self.hstrt_min = None
        self.hstrt_max = None
        self.hstrt_hend_max = None
        self.hend_min = None
        self.hend_max = None
        self.tpfd_min = None
        self.tpfd_max = None

        self.register_commands()

    def register_commands(self) -> None:
        """Register GCode commands."""
        self.gcode.register_command("CHOPPER_TUNE", self.cmd_chopper_tune)
        self.gcode.register_command("CHOPPER_TUNE_DEBUG", self.cmd_chopper_tune_debug)

    def respond_info(self, msg: str) -> None:
        """Respond info through the current GCodeCommand instance.

        Args:
            msg (str): The info message.
        """
        if self._gcmd is None:
            self.gcode.respond_info(msg)
        else:
            self._gcmd.respond_info(msg)

    @property
    def settings(self) -> dict:
        """Return the settings dictionary.

        Returns:
            dict: The settings dictionary.
        """
        if self._settings is None:
            self._settings = self.configfile.get_status(None)["settings"]

        return self._settings

    @property
    def toolhead(self) -> ToolHead:
        """Return the toolhead.

        Returns:
            ToolHead: The toolhead.
        """
        if self._toolhead is None:
            self._toolhead = self.printer.lookup_object("toolhead")
        return self._toolhead

    @property
    def driver_settings(self) -> dict:
        """Return the driver settings dictionary.

        Returns:
            dict: The driver settings dictionary.
        """
        if not self._driver_settings:
            for axis in "xyz":
                driver, _ = self.detect_driver(axis)
                self._driver_settings[f"stepper_{axis}"] = self.settings.get(
                    f"tmc{driver} stepper_{axis}", {}
                )
        return self._driver_settings

    @property
    def stepper_settings(self) -> dict:
        """Return the stepper settings dictionary.

        Returns:
            dict: The stepper settings dictionary.
        """
        if not self._stepper_settings:
            for axis in "xyz":
                self._stepper_settings[f"stepper_{axis}"] = self.settings.get(
                    f"stepper_{axis}", {}
                )
        return self._stepper_settings

    def clean_csv_files(self) -> None:
        """Clean temporary data files and exit."""
        for f in glob.glob(os.path.join(DATA_FOLDER, "*.csv")):
            os.remove(f)

    def detect_driver(self, stepper: str) -> None | tuple[str, str]:
        """Detect the driver of the selected stepper.

        Args:
            stepper (str): The stepper name.

        Returns:
            tuple[None, None] | tuple[str, str]: A tuple containing the stepper
                driver model and sense resistor, or None if the driver thus the
                sense resistor cannot be detected.
        """
        drivers = ["2130", "2208", "2209", "2660", "2240", "5160"]
        stepper = f"stepper_{stepper}"
        resistor = None
        for driver in drivers:
            if "run_current" not in self.settings.get(f"tmc{driver} {stepper}", {}):
                continue
            self.respond_info(f"Selected tmc{driver} for {stepper}")
            if driver != "2240":
                resistor = self.settings[f"tmc{driver} {stepper}"]["sense_resistor"]
            else:
                resistor = self.settings[f"tmc{driver} {stepper}"]["rref"]
            return driver, resistor

        return None, None

    def get_axes_and_steppers(self, axis: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Get main and secondary axis / stepper.

        Args:
            axis (str): The to be tuned.

        Returns:
            tuple[tuple[str, ...], tuple[str, ...]]: A tuple containing:
                - axes (tuple[str, ...]): The main and secondary axis.
                - steppers (tuple[str, ...]): The main and secondary stepper.
        """
        if axis not in ("x", "y", "z"):
            raise self.printer.command_error(f"WARNING!!! Incorrect axis: {axis}")

        if self.kinematics not in ("corexy", "cartesian"):
            raise self.printer.command_error(
                f"WARNING!!! Unsupported kinematics: {self.kinematics}"
            )

        if self.kinematics == "corexy":
            if axis in ("x", "y"):
                if axis == "x":
                    axes = ("x", "y")
                    steppers = ("stepper_x", "stepper_y")
                elif axis == "y":
                    axes = ("y", "x")
                    steppers = ("stepper_y", "stepper_x")
            elif axis == "z":
                axes = ("z", "x")
                steppers = ("stepper_z",)
        elif self.kinematics == "cartesian":
            if axis == "x":
                axes = ("x", "y")
                steppers = ("stepper_x",)
            elif axis == "y":
                axes = ("y", "x")
                steppers = ("stepper_y",)
            elif axis == "z":
                axes = ("z", "x")
                steppers = ("stepper_z",)

        return axes, steppers

    def get_axis_limits(self, axes: list[str]) -> tuple[float, float, float, float]:
        """Select main and secondary axis / stepper.

        Args:
            axes (list[str]): The main and secondary axis.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - a_axis_min (float): The minimum position of the main axis.
                - a_axis_max (float): The maximum position of the main axis.
                - a_axis_mid (float): The middle position of the main axis.
                - b_axis_mid (float): The middle position of the secondary axis.
        """
        if axes[0] == "z":
            a_axis_min = (
                max(self.stepper_settings[f"stepper_{axes[0]}"]["position_min"], 0)
                + self.inset
            )
        else:
            a_axis_min = (
                self.stepper_settings[f"stepper_{axes[0]}"]["position_min"] + self.inset
            )

        a_axis_max = (
            self.stepper_settings[f"stepper_{axes[0]}"]["position_max"] - self.inset
        )
        a_axis_mid = (
            self.stepper_settings[f"stepper_{axes[0]}"]["position_max"]
            + self.stepper_settings[f"stepper_{axes[0]}"]["position_min"]
        ) / 2
        b_axis_mid = (
            self.stepper_settings[f"stepper_{axes[1]}"]["position_max"]
            - self.stepper_settings[f"stepper_{axes[1]}"]["position_min"]
        ) / 2
        return a_axis_min, a_axis_max, a_axis_mid, b_axis_mid

    def get_travel_speed_and_acceleration(self, axes: list[str]) -> tuple[float, float]:
        """Select main and secondary axis / stepper.

        Args:
            axes (list[str]): The main and secondary axis.

        Returns:
            tuple[float, float]: A tuple containing:
                - acceleration (float): The acceleration for the movement.
                - travel_speed (float): The travel speed for idle movements.
        """
        if axes[0] == "z":
            acceleration = self.settings["printer"]["max_z_accel"]
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_z_velocity", 0) / 2 * 60
        else:
            acceleration = self.settings["printer"].get("max_accel")
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_velocity") / 2 * 60

        return acceleration, travel_speed

    def validate_tpfd_values(
        self,
        driver: str,
        tpfd_min: int,
        tpfd_max: int,
    ) -> None:
        """Validate the TPFD min and max values.

        Args:
            driver (str): The stepper driver model.
            tpfd_min (int): The TPFD min value.
            tpfd_max (int): The TPFD max value.
        """
        if tpfd_min != -1 or tpfd_max != -1:
            if driver in ["2240", "5160"]:
                if tpfd_min < 0 or tpfd_max < 0:
                    self.printer.command_error("WARNING!!! Incorrect TPFD values")
            else:
                self.printer.command_error(
                    f"WARNING!!! TMC{driver} don't support register TPFD"
                )

    def reset_registers(self) -> None:
        """Reset registers to default values."""
        # Reset register values
        self.registers = {
            "stepper_count": 0,
            "tbl": -1,
            "toff": -1,
            "hend": -1,
            "hstrt": -1,
            "tpfd": -1,
            "curr": -1,
        }

    def apply_registers(
        self,
        field: str,
        value: int,
        steppers: list[str],
    ) -> None:
        """Apply registers.

        Args:
            field (str): The name of the field to set the value of.
            value (int): The value to set to.
            steppers (list[str]): The name of the steppers to set the register
                field values of.
        """
        if field is None or value is None:
            return

        stepper = steppers[0]  # just update the main stepper
        for stepper_index in range(self.registers["stepper_count"]):
            # stepper_x,
            # stepper_y,
            # stepper_z, stepper_z1, stepper_z2, stepper_z3, ...
            # don't add index for the first stepper
            stepper_index = str(stepper_index) if stepper_index > 0 else ""
            if self.debug:
                self.respond_info(
                    f"Setting {field.lower()} "
                    f"from {self.registers[field]} to {value} "
                    f"on {stepper}{stepper_index}"
                )

            if field.lower() == "curr":
                if self.registers[field.lower()] != value:
                    self.gcode.run_script_from_command(
                        f"SET_TMC_CURRENT STEPPER={stepper} CURRENT={value / 1000}"
                    )
            elif (
                not (field == "tpfd" and value == -1)
                and self.registers[field.lower()] != value
            ):
                self.gcode.run_script_from_command(
                    "SET_TMC_FIELD "
                    f"STEPPER={stepper}{stepper_index} "
                    f"FIELD={field} VALUE={value}"
                )
        # store the last applied value
        self.registers[field.lower()] = value

    def get_stepper_count(self, axis: str) -> int:
        """Get the stepper count of the given axis.

        Args:
            axis (str): One of ["x", "y", "z"].

        Returns:
            int: The stepper count of the requested axis.
        """
        axis_steppers = [
            key for key in self.settings if key.startswith(f"stepper_{axis}")
        ]
        return len(axis_steppers)

    def get_accelerometer_chip(self, accel_chip: str) -> str:
        """Get accelerometer chip.

        Args:
            accel_chip (str): Accelerometer chip name, i.e adxl345.
        """
        # Select accelerometer
        if accel_chip == "default":
            resonance_tester = self.settings.get("resonance_tester", {})
            if "accel_chip" in resonance_tester:
                accel_chip = resonance_tester["accel_chip"]
            else:
                # Use Default accelerometer
                accel_chip = DEFAULT_ACCEL_CHIP

        self.respond_info(f"Selected {accel_chip} for accelerometer")
        return accel_chip

    def get_current_range(
        self,
        measurement_mode: MeasurementMode,
        current_min: int | str,
        current_max: int | str,
        steppers: list[str],
    ) -> tuple[int, int]:
        """Get run current.

        Args:
            measurement_mode (MeasurementMode): The measurement mode.
            current_min (int | str): The minimum current value.
            current_max (int | str): The maximum current value.
            steppers (list[str]): The main and secondary stepper.

        Returns:
            tuple[int, int]: The minimum and maximum current values.
        """
        # Select run_current
        run_current = int(
            float(self.driver_settings[steppers[0]].get("run_current")) * 1000
        )
        if current_min == "default":
            current_min = run_current
            if self.debug:
                self.respond_info(
                    f"Set default run_current: {current_min} mA to run_current_min"
                )
        else:
            current_min = int(current_min)

        if current_max == "default":
            current_max = run_current
            if self.debug:
                self.respond_info(
                    f"Set default run_current: {current_max} mA to run_current_max"
                )
        else:
            current_max = int(current_max)

        if measurement_mode == MeasurementMode.Resonances:
            current_max = current_min

        return current_min, current_max

    def get_default_stepper_parameters(
        self,
        steppers: list[str],
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """Return default stepper parameters.

        Args:
            steppers (list[str]): The main and secondary stepper.

        Returns:
            tuple[int, int, int, int, int, int, int, int]: The default stepper
                parameters.
        """
        tbl_max = tbl_min = self.driver_settings[steppers[0]].get("driver_tbl")
        toff_max = toff_min = self.driver_settings[steppers[0]].get("driver_toff")
        hstrt_max = hstrt_min = self.driver_settings[steppers[0]].get("driver_hstrt")
        hend_max = hend_min = self.driver_settings[steppers[0]].get("driver_hend")

        return (
            tbl_min,
            tbl_max,
            toff_min,
            toff_max,
            hstrt_min,
            hstrt_max,
            hend_min,
            hend_max,
        )

    def configure_speed_limits(
        self,
        min_speed: float,
        max_speed: float,
        speed_change_step: float,
        measure_time: float,
        axes: list[str],
        steppers: list[str],
        a_axis_min: float,
        a_axis_max: float,
        acceleration: float,
    ) -> tuple[float, float, float]:
        """Configure speed limits.

        Args:
            min_speed (float | str): The in speed value, or can be set to
                "default" to auto calculate the value over the required RPM
                value.
            max_speed (float | str): The max speed value, or can be set to
                "default" to auto calculate the value over the required RPM
                value.
            speed_change_step (float | str): The step in each iteration the
                speed will be increased to.
            measure_time (float): The measurement time in seconds.
            axes (list[str]): The main and secondary axis.
            steppers (list[str]): The main and secondary stepper.
            a_axis_min (float): The minimum position of the main axis.
            a_axis_max (float): The maximum position of the main axis.
            acceleration (float): The acceleration for the movement.

        Returns:
            tuple[float, float, float]: The minimum speed, maximum speed and
                speed change step.
        """
        # In vibration measurement mode,
        # search and take registers from printer.cfg,
        # to set the speed range
        if self.measurement_mode == MeasurementMode.Resonances:
            rotation_dist = self.stepper_settings[steppers[0]].get("rotation_distance")
            # get gear ratio
            gear_ratio = self.stepper_settings[steppers[0]].get("gear_ratio")
            if not gear_ratio:  # can be () or None
                gear_ratio = "1:1"
            gear_ratio = tuple(float(r) for r in gear_ratio.split(":"))
            full_steps_per_rotation = self.stepper_settings[steppers[0]].get(
                "full_steps_per_rotation", 200
            )

            steps_multiplier = (
                full_steps_per_rotation
                / 200
                / (float(gear_ratio[0]) / float(gear_ratio[1]))
                * rotation_dist
                / 60
            )

            if min_speed == "default":
                min_speed = float(self.required_rpm[0] * steps_multiplier)
            else:
                min_speed = float(min_speed)

            if max_speed == "default":
                max_required_speed = float(self.required_rpm[1] * steps_multiplier)
                max_speed = min(
                    (
                        (
                            -acceleration * measure_time
                            + (
                                (acceleration * measure_time) ** 2
                                + 4 * acceleration * (a_axis_max - a_axis_min)
                            )
                            ** 0.5
                        )
                        / 2
                    ),
                    max_required_speed,
                )
            else:
                max_speed = float(max_speed)

            if speed_change_step == "default":
                speed_change_step = self.required_rpm[2] * steps_multiplier
            else:
                speed_change_step = float(speed_change_step)
        else:
            # Protect not defined speed & converting str -> float
            if min_speed == "default" or max_speed == "default":
                raise self.printer.command_error(
                    "WARNING!!! Resonance speed must be defined"
                )
            min_speed, max_speed = float(min_speed), float(max_speed)
            speed_change_step = 1

        # Check speed limit
        if axes[0] == "z":
            max_velocity = self.settings["printer"].get("max_z_velocity", 0)
        else:
            max_velocity = self.settings["printer"].get("max_velocity")

        if max_speed > max_velocity:
            raise self.printer.command_error(
                f"WARNING!!! Required speed ({max_speed} mm/s) on axis ({axes[0]}) "
                f"is faster than kinematics allow ({max_velocity}), "
                f"please lower speed or increase speed limit in printer.cfg"
            )

        return min_speed, max_speed, speed_change_step

    def calculate_travel_distance(
        self,
        axes: list[str],
        a_axis_min: float,
        a_axis_max: float,
        max_speed: float,
        acceleration: float,
        measure_time: float,
        travel_distance: float | str,
    ) -> float:
        """Calculate travel distance.

        Args:
            axes (list[str]): The main and secondary axis.
            a_axis_min (float): The minimum position of the main axis.
            a_axis_max (float): The maximum position of the main axis.
            max_speed (float): The maximum speed of the main axis.
            acceleration (float): The acceleration of the main axis.
            measure_time (float): The measurement time in seconds.
            travel_distance (float | str): The travel distance, or can be set
                to "default" to calculate the travel distance with the
                `measure_time`, `max_speed` and `accel_decel_distance`.

        Returns:
            float: The calculated travel distance.
        """
        # Calculate min required toolhead travel distance from speed,
        # acceleration and time
        accel_decel_distance = max_speed**2 / acceleration
        auto_travel_distance = accel_decel_distance + (max_speed * measure_time)
        if self.debug:
            self.respond_info(
                f"Acceleration & deceleration zone = {accel_decel_distance} mm"
            )
            self.respond_info(
                "Auto calculated min required travel distance = "
                f"{auto_travel_distance} mm"
            )

        # Protect exceeding axis limits & calculate travel distance
        if travel_distance == "default":
            if a_axis_min + auto_travel_distance > a_axis_max:
                raise self.printer.command_error(
                    f"WARNING!!! Required travel distance on axis ({axes[0]}) "
                    f"({auto_travel_distance:.2f} mm) is longer than kinematics "
                    "allows, please lower speed or increase acceleration"
                )

            travel_distance = auto_travel_distance
        else:
            travel_distance = int(travel_distance)
            if a_axis_min + travel_distance > a_axis_max:
                travel_distance = a_axis_max - a_axis_min
                if travel_distance < auto_travel_distance:
                    raise self.printer.command_error(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) is "
                        "less than it should be, please increase acceleration "
                        "or lower speed"
                    )
                if travel_distance > auto_travel_distance:
                    self.respond_info(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        "is longer than kinematics allows, lowering..."
                    )
            elif travel_distance < auto_travel_distance:
                travel_distance = auto_travel_distance
                if a_axis_min + auto_travel_distance > a_axis_max:
                    raise self.printer.command_error(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        f"is less than required ({auto_travel_distance:.2f} mm), "
                        "and longer than kinematics allows, please lower "
                        "speed or increase acceleration"
                    )

        return travel_distance

    def display_process_info(
        self,
        current_min: int,
        current_max: int,
        tbl_min: int,
        tbl_max: int,
        toff_min: int,
        toff_max: int,
        hstrt_min: int,
        hstrt_max: int,
        hend_min: int,
        hend_max: int,
        tpfd_min: int,
        tpfd_max: int,
        min_speed: float,
        max_speed: float,
        speed_change_step: float,
        iterations: int,
        search_method: SearchMethod,
        a_axis_min: float,
        travel_distance: float,
    ) -> None:
        """Display process information.

        Args:
            current_min (int): The minimum current value.
            current_max (int): The maximum current value.
            tbl_min (int): The minimum TBL value.
            tbl_max (int): The maximum TBL value.
            toff_min (int): The minimum TOFF value.
            toff_max (int): The maximum TOFF value.
            hstrt_min (int): The minimum HSTRT value.
            hstrt_max (int): The maximum HSTRT value.
            hend_min (int): The minimum HEND value.
            hend_max (int): The maximum HEND value.
            tpfd_min (int): The minimum TPFD value.
            tpfd_max (int): The maximum TPFD value.
            min_speed (float): The minimum speed value.
            max_speed (float): The maximum speed value.
            speed_change_step (float): The speed change step value.
            iterations (int): The number of iterations.
            search_method (SearchMethod): The search method.
            a_axis_min (float): The minimum position of the main axis.
            travel_distance (float): The travel distance value.
        """
        if self.measurement_mode == MeasurementMode.Resonances:
            # Resonance measurement mode uses the minimum values for registers.
            self.respond_info(
                f"Final max travel distance = {travel_distance:.2f} mm, "
                f"position min = {a_axis_min:.2f}, "
                f"traveling = {a_axis_min:.2f} --> {travel_distance + a_axis_min:.2f}"
            )
            self.respond_info(
                f"Start find resonances mode\n"
                f"Method     : {search_method}\n"
                f"speed      : {min_speed:.2f}  --> {max_speed:.2f} mm/s with "
                f"{speed_change_step:.2f} step\n"
                f"current    : {current_min} mA\n"
                f"TBL        : {tbl_min}\n"
                f"TOFF       : {toff_min}\n"
                f"HSTRT      : {hstrt_min}\n"
                f"HEND       : {hend_min}"
            )
        else:
            self.respond_info(
                f"Final travel distance = {travel_distance:.2f} mm, "
                f"position min = {a_axis_min:.2f}, "
                f"traveling = {a_axis_min:.2f} --> {travel_distance + a_axis_min:.2f}"
            )
            self.respond_info(
                "Start of register enumeration mode\n"
                f"Method     : {search_method}\n"
                f"speed      : {min_speed:.2f}  --> {max_speed:.2f}  mm/s\n"
                f"current    : {current_min} --> {current_max} mA\n"
                f"iterations : {iterations}\n"
                f"TBL        : {tbl_min} --> {tbl_max}\n"
                f"TOFF       : {toff_min} --> {toff_max}\n"
                f"HSTRT      : {hstrt_min} --> {hstrt_max}\n"
                f"HEND       : {hend_min} --> {hend_max}\n"
                f"TPFD       : {tpfd_min} --> {tpfd_max}"
            )

    def home(self) -> None:
        """Home."""
        # event_time = self.printer.get_reactor().monotonic()
        # if "xyz" not in self.toolhead.get_status(event_time)["homed_axes"]:
        self.gcode.run_script_from_command("G28 X Y Z")
        self.toolhead.wait_moves()

    def measure_accelerometer_noise(self, accel_chip: str) -> str:
        """Measure accelerometer noise.

        Args:
            accel_chip (str): Accelerometer chip name, i.e adxl345.

        Returns:
            str: The measurement data file path.
        """
        start_time = time.time()
        self.toolhead.wait_moves()
        with AccelerometerMeasure(
            printer=self.printer,
            gcode=self.gcode,
            accel_chip=accel_chip,
            name="stand_still",
        ) as accelerometer_measurement:
            self.gcode.run_script_from_command("G4 P5000")
        # Wait for another 1 second for the whole data to be written
        self.gcode.run_script_from_command("G4 P1000")
        self.toolhead.wait_moves()
        if self.search_method == SearchMethod.BruteForce:
            # move the measurement file to the DATA_FOLDER
            measurement_data_path = accelerometer_measurement.move()
        else:
            # no need to keep the file in adaptive mode so use it from /tmp
            measurement_data_path = accelerometer_measurement.get_full_path()
        self.respond_info(f"Noise Data: {measurement_data_path}")
        if self.debug:
            duration = time.time() - start_time
            self.respond_info(f"AccelerometerMeasure took {duration:0.1f} seconds")
        return measurement_data_path

    def measure_vibrations(
        self,
        coord_generator: CoordGenerator,
        travel_distance: float,
        speed: float,
        accel_chip: str,
        name: str,
    ) -> str:
        """Perform vibration measurement.

        Args:
            coord_generator (CoordGenerator): The coordinate generator.
            travel_distance (float): The travel distance.
            speed (float): The speed for the measurement.
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            name (str): The name of the measurement.

        Returns:
            str: The measurement data file path.
        """
        # Start accel_chip data collection
        with AccelerometerMeasure(
            printer=self.printer,
            gcode=self.gcode,
            accel_chip=accel_chip,
            name=name,
        ) as accelerometer_measurement:
            # go in both directions at once
            next_coord = coord_generator.next(travel_distance)
            self.gcode.run_script_from_command(
                "G0 "
                f"X{next_coord.x:0.2f} "
                f"Y{next_coord.y:0.2f} "
                f"Z{next_coord.z:0.2f} "
                f"F{speed * 60}"
            )
        # Move to the initial position
        self.gcode.run_script_from_command(
            "G0 "
            f"X{self.initial_position.x:0.2f} "
            f"Y{self.initial_position.y:0.2f} "
            f"Z{self.initial_position.z:0.2f} "
            f"F{self.travel_speed}"
        )
        coord_generator.current_coord.x = self.initial_position.x
        coord_generator.current_coord.y = self.initial_position.y
        coord_generator.current_coord.z = self.initial_position.z
        self.toolhead.wait_moves()

        if self.search_method == SearchMethod.BruteForce:
            # move the measurement file to the DATA_FOLDER
            measurement_data_path = accelerometer_measurement.move()
        else:
            # no need to keep the file in adaptive mode so use it from /tmp
            measurement_data_path = accelerometer_measurement.get_full_path()
        # self.respond_info(f"Accel. data: {measurement_data_path}")

        self.number_of_samples += 1

        return measurement_data_path

    def calculate_frequency(self, tbl: int, toff: int) -> float:
        """Calculate frequency based on TBL and TOFF values.

        Args:
            tbl (int): The TBL value.
            toff (int): The TOFF value.

        Returns:
            float: The calculated frequency.
        """
        return 1 / (
            2 * (12 + 32 * toff) * 1 / (1000000 * self.fclk)
            + 2 * 1 / (1000000 * self.fclk) * 16 * (1.5**tbl)
        )

    def get_initial_direction(self, axes: list[str]) -> Coord:
        """Return the initial direction based on the axes and kinematics.

        Args:
            axes (list[str]): The main and secondary axis.

        Returns:
            Coord: The initial direction.
        """
        initial_direction = Coord((1, 0, 0))
        if axes[0] == "x":
            if self.kinematics == "corexy":
                initial_direction = Coord((1, 1, 0)).unitize()
            else:
                initial_direction = Coord((1, 0, 0))
        elif axes[0] == "y":
            if self.kinematics == "corexy":
                initial_direction = Coord((1, -1, 0)).unitize()
            else:
                initial_direction = Coord((0, 1, 0))
        elif axes[0] == "z":
            initial_direction = Coord((0, 0, 1))
        return initial_direction

    def chopper_tune(
        self,
        axis: str,
        current_min: int | str = "default",
        current_max: int | str = "default",
        tbl_min: int = 0,
        tbl_max: int = 3,
        toff_min: int = 1,
        toff_max: int = 8,
        hstrt_hend_max: int = 16,
        hstrt_min: int = 0,
        hstrt_max: int = 7,
        hend_min: int = 2,
        hend_max: int = 15,
        tpfd_min: int = -1,
        tpfd_max: int = -1,
        min_speed: int | str = "default",
        max_speed: int | str = "default",
        speed_change_step: int | str = "default",
        search_method: SearchMethod = SearchMethod.BruteForce,
        travel_distance: int | str = "default",
        direction: int = 1,
        accel_chip: str = "default",
        run_plotter: bool = True,
    ) -> None | dict:
        """Measure vibrations and tune stepper motors for low noise.

        Args:
            axis (str): Axis to tune. Should be one of ["x", "y", "z"].
            current_min (int | str): Minimum steeper current in mA, or use
                "default" to set the current to the `run_current` value.
            current_max (int | str): Maximum steeper current in mA, or use
                "default" to set the current to the `run_current` value.
            tbl_min (int): The min TBL value.
            tbl_max (int): The max TBL value.
            toff_min (int): The min TOFF value.
            toff_max (int): The max TOFF value.
            hstrt_hend_max (int): The max HSTRT_HEND value
            hstrt_min (int): The min HSTRT value.
            hstrt_max (int): The max HSTRT value.
            hend_min (int): The min HEND value.
            hend_max (int): The max HEND value.
            tpfd_min (int): The min TPFD value.
            tpfd_max (int): The max TPFD value.
            min_speed (int | str): The in speed value, or can be set to
                "default" to auto calculate the value over the required RPM
                value.
            max_speed (int | str): The max speed value, or can be set to
                "default" to auto calculate the value over the required RPM
                value.
            speed_change_step (int | str): The step in each iteration the speed
                will be increased to.
            search_method (SearchMethod): The search method, can be one of
                [SearchMethod.BruteForce, SearchMethod.Adaptive], default value
                is SearchMethod.BruteForce.
            travel_distance (int | str): The travel distance, or can be set to
                "default" to calculate the travel distance with the
                `measure_time`, `max_speed` and `accel_decel_distance`.
            direction (int): The movement direction, can be 1 or -1.
                1 means starting from the minimum position to maximum position,
                -1 means starting from the maximum position to minimum position.
            accel_chip (str): The name of the acceleration chip.
            run_plotter (bool): If set to True, the magnitude graphs will be
                generated after the vibration measurements are completed.

        Returns:
            None | dict: The best parameters found, or None if tuning was not
                done using the adaptive method.
        """
        self.search_method = search_method

        # Force brute_force in vibration measurement mode
        if self.measurement_mode == MeasurementMode.Resonances:
            self.search_method = SearchMethod.BruteForce

        self.respond_info(f"Selected {self.search_method} as search method")

        measure_time = self.measure_time / 1000
        self.reset_registers()
        # Find the steppers count of the main axis
        self.registers["stepper_count"] = self.get_stepper_count(axis)

        self.driver, self.sense_resistor = self.detect_driver(stepper=axis)
        self.validate_tpfd_values(self.driver, tpfd_min, tpfd_max)

        axes, steppers = self.get_axes_and_steppers(axis)

        a_axis_min, a_axis_max, a_axis_mid, b_axis_mid = self.get_axis_limits(axes)
        acceleration, self.travel_speed = self.get_travel_speed_and_acceleration(axes)
        accel_chip = self.get_accelerometer_chip(accel_chip)

        current_min, current_max = self.get_current_range(
            self.measurement_mode, current_min, current_max, steppers
        )

        if self.measurement_mode == MeasurementMode.Resonances:
            # In vibration measurement mode,
            # search and take registers from printer.cfg
            (
                tbl_min,
                tbl_max,
                toff_min,
                toff_max,
                hstrt_min,
                hstrt_max,
                hend_min,
                hend_max,
            ) = self.get_default_stepper_parameters(steppers)

        (min_speed, max_speed, speed_change_step) = self.configure_speed_limits(
            min_speed,
            max_speed,
            speed_change_step,
            measure_time,
            axes,
            steppers,
            a_axis_min,
            a_axis_max,
            acceleration,
        )

        travel_distance = self.calculate_travel_distance(
            axes,
            a_axis_min,
            a_axis_max,
            max_speed,
            acceleration,
            measure_time,
            travel_distance,
        )

        # Info message
        self.display_process_info(
            current_min,
            current_max,
            tbl_min,
            tbl_max,
            toff_min,
            toff_max,
            hstrt_min,
            hstrt_max,
            hend_min,
            hend_max,
            tpfd_min,
            tpfd_max,
            min_speed,
            max_speed,
            speed_change_step,
            self.iterations,
            self.search_method,
            a_axis_min,
            travel_distance,
        )

        # Home regardless of previous homing state
        self.home()
        home_pos = Coord(self.toolhead.get_position())

        # Get initial position and direction
        self.initial_position = {
            "x": Coord((a_axis_mid, b_axis_mid, home_pos.z)),
            "y": Coord((b_axis_mid, a_axis_mid, home_pos.z)),
            "z": Coord((b_axis_mid, home_pos.y, a_axis_mid)),
        }[axes[0]]
        self.initial_direction = self.get_initial_direction(axes)
        if direction == -1:
            self.initial_direction = self.initial_direction * -1

        # if this is not running in "find resonances" mode,
        # move away from the middle exactly half or a travel distance
        # if measurement_mode == MeasurementMode.Vibrations:
        self.initial_position -= self.initial_direction * (travel_distance / 2)

        self.gcode.run_script_from_command(f"SET_VELOCITY_LIMIT ACCEL={acceleration}")
        self.gcode.run_script_from_command(
            f"SET_VELOCITY_LIMIT ACCEL_TO_DECEL={acceleration}"
        )
        # Move to the initial position
        self.gcode.run_script_from_command(
            "G0 "
            f"X{self.initial_position.x:0.2f} "
            f"Y{self.initial_position.y:0.2f} "
            f"Z{self.initial_position.z:0.2f} "
            f"F{self.travel_speed}"
        )

        # Clean csv files while going to the initial position
        self.clean_csv_files()

        # Wait for move to complete
        self.toolhead.wait_moves()

        # Measure accelerometer noise
        static_data_path = self.measure_accelerometer_noise(accel_chip)
        static_noise_magnitude = tuple(calc_static_magnitude(static_data_path))

        # Create the coordinate generator
        coord_generator = CoordGenerator(
            direction=self.initial_direction,
            start_coord=self.initial_position,
        )

        # calculated vars
        self.speed = min_speed
        self.max_speed = max_speed
        self.travel_distance = travel_distance
        self.coord_generator = coord_generator
        self.accel_chip = accel_chip
        self.steppers = steppers
        self.current = current_min
        self.static_noise_magnitude = static_noise_magnitude  # make it immutable

        # bounds
        self.current_min = current_min
        self.current_max = current_max

        self.tbl_min = tbl_min
        self.tbl_max = tbl_max
        self.toff_min = toff_min
        self.toff_max = toff_max
        self.hstrt_min = hstrt_min
        self.hstrt_max = hstrt_max
        self.hstrt_hend_max = hstrt_hend_max
        self.hend_min = hend_min
        self.hend_max = hend_max
        self.tpfd_min = tpfd_min
        self.tpfd_max = tpfd_max

        self.bounds = [
            (self.current_min, self.current_max),
            (self.tbl_min, self.tbl_max),
            (self.toff_min, self.toff_max),
            (self.hstrt_min, self.hstrt_max),
            (self.hend_min, self.hend_max),
            (self.tpfd_min, self.tpfd_max),
        ]
        best_parameters = None
        if self.search_method == SearchMethod.Adaptive:
            # Run adaptive optimization
            best_parameters = self.run_optimization()
        else:
            # Brute-force search
            speed_vs_vibrations = []
            for current in range(
                current_min, current_max + 1, self.current_change_step
            ):
                for tbl in range(tbl_min, tbl_max + 1):
                    for toff in range(toff_min, toff_max + 1):
                        for hstrt in range(hstrt_min, hstrt_max + 1):
                            for hend in range(hend_min, hend_max + 1):
                                if (hend + hstrt) > hstrt_hend_max:
                                    continue
                                for tpfd in range(tpfd_min, tpfd_max + 1):
                                    for speed in range(
                                        int(min_speed * 100),
                                        int(max_speed * 100) + 1,
                                        int(speed_change_step * 100),
                                    ):
                                        speed = speed / 100
                                        for iteration in range(self.iterations):
                                            measured_vibrations = (
                                                self.execute_vibration_measurement(
                                                    speed,
                                                    max_speed,
                                                    travel_distance,
                                                    coord_generator,
                                                    accel_chip,
                                                    steppers,
                                                    static_noise_magnitude,
                                                    iteration,
                                                    current,
                                                    tbl,
                                                    toff,
                                                    hstrt,
                                                    hend,
                                                    tpfd,
                                                )
                                            )
                                            speed_vs_vibrations.append(
                                                (speed, measured_vibrations)
                                            )
            if self.measurement_mode == MeasurementMode.Resonances:
                max_vibrations_and_speed = sorted(
                    speed_vs_vibrations, key=lambda x: x[1]
                )[-1]
                self.respond_info(
                    "Max vibrations seems to be at "
                    f"{max_vibrations_and_speed[0]:0.2f} mm/s"
                )

        self.gcode.run_script_from_command("G4 P500")
        self.gcode.run_script_from_command(
            f"G0 {axis}{a_axis_mid} F{self.travel_speed}"
        )
        self.toolhead.wait_moves()
        if self.search_method != SearchMethod.Adaptive:
            if run_plotter:
                self.respond_info("Magnitude graphs generation...")
                self.respond_info("This may take a while, please wait")
                # export data to processing
                self.gcode.run_script_from_command(
                    "RUN_SHELL_COMMAND CMD=chop_tune "
                    f"PARAMS='iterations={self.iterations} "
                    f"driver={self.driver} "
                    f"sense_resistor={self.sense_resistor}'"
                )
            # output data info
            self.respond_info(
                "To run parser manually; type - "
                "RUN_SHELL_COMMAND CMD=chop_tune "
                f"PARAMS='iterations={self.iterations} "
                f"driver={self.driver} "
                f"sense_resistor={self.sense_resistor}"
            )

        # reset number of samples
        self.number_of_samples = 0
        return best_parameters

    @cache
    def execute_vibration_measurement(
        self,
        speed: float,
        max_speed: float,
        travel_distance: float,
        coord_generator: CoordGenerator,
        accel_chip: str,
        steppers: tuple[str, ...],
        static_noise_magnitude: float,
        iteration: int,
        current: int,
        tbl: int,
        toff: int,
        hstrt: int,
        hend: int,
        tpfd: int,
    ) -> float:
        """Execute a single vibration measurement.

        Args:
            speed (float): The speed for the measurement.
            max_speed (float): The maximum speed for the measurement.
            travel_distance (float): The travel distance.
            coord_generator (CoordGenerator): The coordinate generator.
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            steppers (list[str]): The main and secondary stepper.
            static_noise_magnitude (float): The static noise magnitude.
            iteration (int): The current iteration number.
            current (int): The current value.
            tbl (int): The TBL value.
            toff (int): The TOFF value.
            hstrt (int): The HSTRT value.
            hend (int): The HEND value.
            tpfd (int): The TPFD value.

        Returns:
            float: The measured vibrations.
        """
        # Set tbl values
        # Set toff values
        # Set hend, and hstrt values
        self.apply_registers(steppers=steppers, field="curr", value=current)
        self.apply_registers(steppers=steppers, field="tbl", value=tbl)
        self.apply_registers(steppers=steppers, field="toff", value=toff)
        self.apply_registers(steppers=steppers, field="hend", value=hend)
        self.apply_registers(steppers=steppers, field="hstrt", value=hstrt)
        self.apply_registers(steppers=steppers, field="tpfd", value=tpfd)

        # Dump TMC settings
        self.gcode.run_script_from_command(
            f"DUMP_TMC STEPPER={steppers[0]} REGISTER=chopconf"
        )
        freq = self.calculate_frequency(tbl, toff)
        name = (
            f"__{current}_{tbl}_{toff}_{hstrt}_"
            f"{hend}_{tpfd}_{speed * 100:.0f}_"
            f"{freq:.0f}_{iteration + 1}__"
        )

        real_travel_distance = travel_distance
        if self.measurement_mode == MeasurementMode.Resonances:
            # when finding resonances,
            # keep the travel duration constant
            real_travel_distance = travel_distance * (speed / max_speed)
            self.respond_info(
                f"Speed {speed:0.2f} mm/s on {real_travel_distance:0.2f} mm"
            )
        self.toolhead.wait_moves()

        measurement_data_path = self.measure_vibrations(
            coord_generator,
            real_travel_distance,
            speed,
            accel_chip,
            name,
        )

        # measured_vibrations should be used to optimize the inputs with scipy.optimize
        measured_vibrations = calc_magnitude(
            data_path=measurement_data_path, static_data=static_noise_magnitude
        )
        if self.search_method == SearchMethod.Adaptive:
            os.remove(measurement_data_path)  # no need to keep the file

        self.respond_info(f"Measured vibrations: {measured_vibrations:0.2f} mm/s²")
        return measured_vibrations

    def objective_function(self, params: list[float]) -> float:
        """Objective function for optimization.

        Args:
            params (list[float]): The parameters to optimize.

        Returns:
            float: The average measured vibrations.
        """
        current, tbl, toff, hstrt, hend, tpfd = [round(p) for p in params]

        # penalize hstart + hend > hstrt_hend_max
        if hstrt + hend > self.hstrt_hend_max:
            self.respond_info(
                f"Penalizing hstrt + hend > {self.hstrt_hend_max}: inf mm/s²"
            )
            return float('inf')

        total_vibrations = 0
        for iteration in range(self.iterations):
            measured_vibrations = self.execute_vibration_measurement(
                self.speed,
                self.max_speed,
                self.travel_distance,
                self.coord_generator,
                self.accel_chip,
                self.steppers,
                self.static_noise_magnitude,
                iteration,
                current,
                tbl,
                toff,
                hstrt,
                hend,
                tpfd,
            )
            total_vibrations += measured_vibrations
        total_vibrations /= self.iterations
        self.respond_info(f"Mean vibrations: {total_vibrations:0.2f} mm/s²")
        return total_vibrations

    def run_optimization(self) -> list[int]:
        """Run the optimization process.

        Returns:
            dict: The best parameters found.
        """
        # TODO: Use a for-loop to do multi-step optimization with narrowed bounds
        self.respond_info("Starting optimization 1/2...")
        start_time = time.time()
        overall_start_time = start_time

        # 'strategy' and 'popsize' are tuned to reduce total measurements
        # 'tol' can be higher since our parameters are discrete

        # Do a two-step optimization to speed up the process
        # First run, lock all params except toff and hend
        partial_bounds = [
            (self.current_min, self.current_min),  # lock current
            (self.tbl_min, self.tbl_min),  # lock current
            (self.toff_min, self.toff_max),
            (self.hstrt_min, self.hstrt_min),  # lock current
            (self.hend_min, self.hend_max),
            (self.tpfd_min, self.tpfd_min),  # lock current
        ]

        # set initial values
        self.apply_registers(steppers=self.steppers, field="curr", value=self.current_min)
        self.apply_registers(steppers=self.steppers, field="tbl", value=self.tbl_min)
        self.apply_registers(steppers=self.steppers, field="toff", value=self.toff_min)
        self.apply_registers(steppers=self.steppers, field="hend", value=self.hend_min)
        self.apply_registers(steppers=self.steppers, field="hstrt", value=self.hstrt_min)
        self.apply_registers(steppers=self.steppers, field="tpfd", value=self.tpfd_min)

        result = differential_evolution(
            self.objective_function,
            partial_bounds,
            init="sobol",
            strategy="best1bin",
            maxiter=10,
            popsize=5,  # Total evaluations = maxiter * popsize * N_params
            tol=0.1,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,  # Polish uses local minimize, which we avoid for discrete
        )

        best_params = [round(p) for p in result.x]
        duration = time.time() - start_time
        first_run_duration = duration

        overall_best_params = {
            "current": best_params[0],
            "tbl": best_params[1],
            "toff": best_params[2],
            "hstrt": best_params[3],
            "hend": best_params[4],
            "tpfd": best_params[5],
        }

        self.respond_info(
            f"Optimization 1/2 Completed in {first_run_duration:.2f} seconds!\n"
            f"Number of samples : {self.number_of_samples}\n"
            f"Best Score        : {result.fun:.2f}\n\n"
            "Parameters\n"
            "----------\n"
            f"current      : {overall_best_params['current']}\n"
            f"driver_tbl   : {overall_best_params['tbl']}\n"
            f"driver_toff  : {overall_best_params['toff']}\n"
            f"driver_hstrt : {overall_best_params['hstrt']}\n"
            f"driver_hend  : {overall_best_params['hend']}\n"
        )
        if self.driver in ["2240", "5160"]:
            self.respond_info(f"driver_tpfd : {overall_best_params['tpfd']}")

        # Second run, lock toff and hend to best values from first run
        start_time = time.time()
        self.respond_info(
            "Starting optimization 2/2 with narrowed bounds..."
        )

        partial_bounds = [
            (self.current_min, self.current_max),
            (self.tbl_min, self.tbl_max),
            (overall_best_params["toff"], overall_best_params["toff"]),  # lock to the best found
            (self.hstrt_min, self.hstrt_max),
            (overall_best_params["hend"], overall_best_params["hend"]),  # lock to the best found
            (self.tpfd_min, self.tpfd_max),
        ]

        result = differential_evolution(
            self.objective_function,
            partial_bounds,
            init="sobol",
            strategy="best1bin",
            maxiter=10,
            popsize=5,  # Total evaluations = maxiter * popsize * N_params
            tol=0.1,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=False,  # Polish uses local minimize, which we avoid for discrete
        )

        best_params = [round(p) for p in result.x]

        overall_best_params.update({
            "current": best_params[0],
            "tbl": best_params[1],
            # "toff": best_params[2],
            "hstrt": best_params[3],
            # "hend": best_params[4],
            "tpfd": best_params[5],
        })

        second_run_duration = time.time() - start_time
        overall_duration = time.time() - overall_start_time

        self.respond_info(
            f"Optimization Completed in {overall_duration:.2f} seconds!\n"
            f"Stage 1/2 took {first_run_duration:.2f} seconds!\n"
            f"Stage 2/2 took {second_run_duration:.2f} seconds!\n"
            f"Number of samples : {self.number_of_samples}\n"
            f"Best Score        : {result.fun:.2f}\n\n"
            "Parameters\n"
            "----------\n"
            f"current      : {overall_best_params['current']}\n"
            f"driver_tbl   : {overall_best_params['tbl']}\n"
            f"driver_toff  : {overall_best_params['toff']}\n"
            f"driver_hstrt : {overall_best_params['hstrt']}\n"
            f"driver_hend  : {overall_best_params['hend']}\n"
        )
        if self.driver in ["2240", "5160"]:
            self.respond_info(f"driver_tpfd : {overall_best_params['tpfd']}")

        # clear the cache for the next runs
        self.execute_vibration_measurement.cache_clear()

        return overall_best_params

    def save_configs(self, best_parameters: dict | None) -> None:
        """Save the best parameters to printer.cfg.

        Args:
            best_parameters (dict | None): The best parameters found.
        """
        if best_parameters is None:
            return

        for field, value in best_parameters.items():
            if field == "tpfd" and self.driver not in ["2240", "5160"]:
                continue  # skip tpfd for unsupported drivers

            field_name = f"driver_{field}" if field != "current" else "run_current"
            value = str(value) if field != "current" else f"{value / 1000:0.2f}"

            self.configfile.set(
                f"tmc{self.driver} {self.steppers[0]}",
                field_name,
                value,
            )

        self.respond_info(
            "Best parameters saved to printer.cfg, run SAVE_CONFIG to apply."
        )

    @gcmd_grabber
    def cmd_chopper_tune(self, gcmd: GCodeCommand) -> bool:
        """Tune stepper values.

        Args:
            gcmd (GCodeCommand): The G-Code command.

        Returns:
            bool: True if command completed successfully, False otherwise.
        """
        try:
            axis = gcmd.get("AXIS", "x").lower()
            current_min = gcmd.get("CURRENT_MIN_MA", "default").lower()
            current_max = gcmd.get("CURRENT_MAX_MA", "default").lower()
            tbl_min = int(gcmd.get("TBL_MIN", 0))
            tbl_max = int(gcmd.get("TBL_MAX", 3))
            toff_min = int(gcmd.get("TOFF_MIN", 1))
            toff_max = int(gcmd.get("TOFF_MAX", 8))
            hstrt_hend_max = int(gcmd.get("HSTRT_HEND_MAX", 16))
            hstrt_min = int(gcmd.get("HSTRT_MIN", 0))
            hstrt_max = int(gcmd.get("HSTRT_MAX", 7))
            hend_min = int(gcmd.get("HEND_MIN", 2))
            hend_max = int(gcmd.get("HEND_MAX", 15))
            tpfd_min = int(gcmd.get("TPFD_MIN", -1))
            tpfd_max = int(gcmd.get("TPFD_MAX", -1))
            min_speed = gcmd.get("MIN_SPEED", "default").lower()
            search_method = SearchMethod.to_method(
                gcmd.get("SEARCH_METHOD", "brute_force").lower()
            )
            # search_method can be brute_force or adaptive
            if IS_DIGIT.match(min_speed):
                min_speed = float(min_speed)
            max_speed = gcmd.get("MAX_SPEED", "default").lower()
            if IS_DIGIT.match(max_speed):
                min_speed = float(max_speed)
            speed_change_step = gcmd.get("SPEED_CHANGE_STEP", "default").lower()
            if IS_DIGIT.match(speed_change_step):
                speed_change_step = float(speed_change_step)
            self.iterations = int(gcmd.get("ITERATIONS", 1))
            direction = int(gcmd.get("DIRECTION", 1))
            travel_distance = gcmd.get("TRAVEL_DISTANCE", "default").lower()
            if IS_DIGIT.match(travel_distance):
                travel_distance = float(travel_distance)
            accel_chip = gcmd.get("ACCELEROMETER", "default").lower()

            self.measurement_mode = {
                "0": MeasurementMode.Vibrations,
                "1": MeasurementMode.Resonances,
                "false": MeasurementMode.Vibrations,
                "true": MeasurementMode.Resonances,
            }.get(
                gcmd.get("FIND_RESONANCES", "false").lower(), MeasurementMode.Resonances
            )
            run_plotter = {
                "0": False,
                "1": True,
                "false": False,
                "true": True,
            }.get(gcmd.get("RUN_PLOTTER", "true").lower(), True)

            return self.chopper_tune(
                axis=axis,
                current_min=current_min,
                current_max=current_max,
                tbl_min=tbl_min,
                tbl_max=tbl_max,
                toff_min=toff_min,
                toff_max=toff_max,
                hstrt_hend_max=hstrt_hend_max,
                hstrt_min=hstrt_min,
                hstrt_max=hstrt_max,
                hend_min=hend_min,
                hend_max=hend_max,
                tpfd_min=tpfd_min,
                tpfd_max=tpfd_max,
                min_speed=min_speed,
                max_speed=max_speed,
                speed_change_step=speed_change_step,
                search_method=search_method,
                travel_distance=travel_distance,
                direction=direction,
                accel_chip=accel_chip,
                run_plotter=run_plotter,
            )
        except Exception:
            self.respond_info(traceback.format_exc())

    @gcmd_grabber
    def cmd_chopper_tune_debug(self, gcmd: GCodeCommand) -> bool:
        """Development debug tool.

        Args:
            gcmd (GCodeCommand): The G-Code command.

        Returns:
            bool: True if command completed successfully, False otherwise.
        """
        try:
            # self.respond_info(f"required_rpm: {self.required_rpm}")
            self.respond_info(f"x stepper count: {self.get_stepper_count('x')}")
            self.respond_info(f"y stepper count: {self.get_stepper_count('y')}")
            self.respond_info(f"z stepper count: {self.get_stepper_count('z')}")
        except Exception as e:
            self.respond_info(traceback.format_exc(e))


def load_config(config: ConfigWrapper) -> ChopperTune:
    """Load the ChopperTune config prefix.

    Args:
        config (ConfigWrapper): The config wrapper.

    Returns:
        ChopperTune: The ChopperTune instance.
    """
    return ChopperTune(config)
