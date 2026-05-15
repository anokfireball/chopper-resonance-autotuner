"""Chopper Tune extension for Klipper.

TMC drivers registers calibration tool.

Copyright (C) 2024  Alexander Fedorov <altzbox@gmail.com>
Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>

This file may be distributed under the terms of the GNU GPLv3 license.
"""

# Standard Library Imports
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING

# Third Party Imports
import numpy as np
from scipy import signal
from scipy.optimize import brute, differential_evolution

# Klipper Imports
import gcode

if TYPE_CHECKING:
    import sys
    from types import TracebackType

    from configfile import ConfigWrapper
    from extras.adxl345 import ADXL345, Accel_Measurement
    from extras.lis2dw import LIS2DW
    from gcode import GCodeCommand, GCodeDispatch
    from klippy import Printer
    from reactor import PollReactor
    from toolhead import ToolHead

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self


DEFAULT_ACCEL_CHIP = "adxl345"
RESULTS_FOLDER = os.path.expanduser("~/printer_data/config/chopper_magnitude")
DATA_FOLDER = os.path.expanduser("~/printer_data/config/chopper_magnitude/tmp")

FCLK = 12  # MHz
CUTOFF_RANGE = 5
# Graphs generation
COLORS = [
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

SUPPORTED_DRIVERS = ["2130", "2208", "2209", "2660", "2240", "5160"]
DRIVERS_SUPPORTING_TPFD = ["2240", "5160"]

DEFAULT_REGISTER_VALUES = {
    "TBL": (0, 3),
    "TOFF": (1, 8),
    "EXTRA_HYSTERESIS": (0, 8),
    "TPFD": (0, 4),
}


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
    Progressive = 2

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
        accelerometer: ADXL345 | LIS2DW,
    ) -> None:
        self.accelerometer = accelerometer
        self.bg_client = None
        self.samples: None | list[Accel_Measurement] = None

    def __enter__(self) -> Self:
        """Enter to the context."""
        if self.bg_client is None:
            self.bg_client = self.accelerometer.start_internal_client()
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
        self.bg_client.finish_measurements()
        # retrieve data
        self.samples = self.bg_client.samples or self.bg_client.get_samples()


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


class ChopperTune:
    """The main class to handle the chopper tune functionality.

    Args:
        config (ConfigWrapper): The configuration wrapper.
    """

    def __init__(self, config: ConfigWrapper) -> None:
        self.printer: Printer = config.get_printer()
        self.gcode: GCodeDispatch = self.printer.lookup_object("gcode")
        self.configfile = self.printer.lookup_object("configfile")
        self.toolhead: None | ToolHead = None
        self.accelerometer: None | ADXL345 | LIS2DW = None
        self.settings = None
        self.reactor: PollReactor = self.printer.get_reactor()
        self.driver_settings = {}
        self.stepper_settings = {}
        self.registers = {
            "stepper_count": 0,
            "tbl": -1,
            "toff": -1,
            "extra_hysteresis": -1,
        }

        # config values
        self.debug = config.getboolean("debug", False)
        self.inset = config.getfloat("inset", 10)
        self.measure_time = config.getint("measure_time", 1250)
        self.required_rpm = config.getfloatlist("required_rpm", [37.5, 150, 1.5])
        self.delay = config.getfloat("delay", 500)

        self.kinematics = config.getsection("printer").get("kinematics")

        # runtime variables
        self.driver = None
        self.resistor = None
        self.samples = {}  # store all samples
        self.total_expected_samples = -1  # the expected number of samples
        self.number_of_samples = 0  # current number of samples taken
        self.number_of_real_samples = 0  # current number of real samples taken
        self.best_result = 999_999_999  # store the best result
        self.speed_vs_vibrations = []  # store speed vs vibrations
        self.global_start_time = 0  # store the global start time

        # Calculated values
        self.search_method = None
        self.measurement_mode = MeasurementMode.Vibrations
        self.travel_speed = None
        self.travel_distance = None
        self.coord_generator = None
        self.accel_chip_name = None
        self.steppers = None
        self.static_acceleration_vector = None
        self.static_acceleration_magnitude = None
        self.initial_position = None
        self.initial_direction = None

        # Bounds
        self.bounds = []
        self.extra_hyst_min = None
        self.extra_hyst_max = None
        self.tbl_min = None
        self.tbl_max = None
        self.toff_min = None
        self.toff_max = None
        self.min_speed = None
        self.max_speed = None
        self.speed_change_step = None

        self.register_commands()
        self.printer.register_event_handler("klippy:connect", self._connect)

    def register_commands(self) -> None:
        """Register GCode commands."""
        self.gcode.register_command("CHOPPER_TUNE", self.cmd_chopper_tune)
        self.gcode.register_command("CHOPPER_TUNE_DEBUG", self.cmd_chopper_tune_debug)

    def _connect(self) -> None:
        """Handle printer connect event."""
        self.settings = self.configfile.get_status(None)["settings"]
        self.toolhead = self.printer.lookup_object("toolhead")
        for axis in "xyz":
            driver, _ = self.detect_driver(axis)
            self.driver_settings[f"stepper_{axis}"] = self.settings.get(
                f"tmc{driver} stepper_{axis}", {}
            )
            self.stepper_settings[f"stepper_{axis}"] = self.settings.get(
                f"stepper_{axis}", {}
            )

    def detect_driver(self, stepper: str) -> None | tuple[str, str]:
        """Detect the driver of the selected stepper.

        Args:
            stepper (str): The stepper name.

        Returns:
            tuple[None, None] | tuple[str, str]: A tuple containing the stepper
                driver model and sense resistor, or None if the driver thus the
                sense resistor cannot be detected.
        """
        drivers = SUPPORTED_DRIVERS
        stepper = f"stepper_{stepper}"
        resistor = None
        for driver in drivers:
            if "run_current" not in self.settings.get(f"tmc{driver} {stepper}", {}):
                continue
            self.gcode.respond_info(f"Selected tmc{driver} for {stepper}")
            if driver != "2240":
                resistor = self.settings[f"tmc{driver} {stepper}"]["sense_resistor"]
            else:
                resistor = self.settings[f"tmc{driver} {stepper}"]["rref"]
            return driver, resistor

        return None, None

    def get_axes_and_steppers(
        self, axis: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
            + self.stepper_settings[f"stepper_{axes[1]}"]["position_min"]
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
            travel_speed = self.settings["printer"].get("max_z_velocity", 0) / 2
        else:
            acceleration = self.settings["printer"].get("max_accel")
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_velocity") / 2

        return acceleration, travel_speed

    def reset_sample_data(self) -> None:
        """Reset sample data."""
        self.total_expected_samples = -1
        self.number_of_samples = 0
        self.number_of_real_samples = 0
        self.best_result = 999_999_999
        self.samples = {}
        self.speed_vs_vibrations = []
        self.global_start_time = 0

    def reset_registers(self) -> None:
        """Reset registers to default values."""
        # Reset register values
        self.registers = {
            "stepper_count": 0,
            "tbl": -1,
            "toff": -1,
            "extra_hysteresis": -1,
        }

    def apply_autotune(
        self,
        steppers: list[str],
        extra_hysteresis: int,
        tbl: int,
        toff: int,
        tpfd: int = None,
    ) -> None:
        """Apply autotune parameters via AUTOTUNE_TMC command.

        Args:
            steppers (list[str]): The steppers to apply parameters to.
            extra_hysteresis (int): The extra hysteresis value.
            tbl (int): The TBL value.
            toff (int): The TOFF value.
            tpfd (int): The TPFD value (optional, for supported drivers).
        """
        if (
            self.registers["extra_hysteresis"] == extra_hysteresis
            and self.registers["tbl"] == tbl
            and self.registers["toff"] == toff
            and self.registers.get("tpfd") == tpfd
        ):
            return  # no change needed

        stepper = steppers[0]
        gcode_cmd = (
            f"AUTOTUNE_TMC STEPPER={stepper} "
            f"EXTRA_HYSTERESIS={extra_hysteresis} "
            f"TBL={tbl} TOFF={toff}"
        )
        if tpfd is not None and self.driver_type in ["tmc2240", "tmc5160"]:
            gcode_cmd += f" TPFD={tpfd}"

        if self.debug:
            self.gcode.respond_info(gcode_cmd)
        self.gcode.run_script_from_command(gcode_cmd)
        
        self.registers["extra_hysteresis"] = extra_hysteresis
        self.registers["tbl"] = tbl
        self.registers["toff"] = toff
        if tpfd is not None:
            self.registers["tpfd"] = tpfd

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

    def get_accelerometer_chip(self, accel_chip_name: str, axis: None | str) -> str:
        """Get accelerometer chip.

        Args:
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            axis (None | str): The selected axis.
        """
        # Select accelerometer
        if accel_chip_name == "default":
            resonance_tester = self.settings.get("resonance_tester", {})
            if "accel_chip" in resonance_tester:
                accel_chip_name = resonance_tester["accel_chip"]
            elif axis is not None and f"accel_chip_{axis}" in resonance_tester:
                accel_chip_name = resonance_tester[f"accel_chip_{axis}"]
            else:
                # Use Default accelerometer
                accel_chip_name = DEFAULT_ACCEL_CHIP

        self.gcode.respond_info(f"Selected {accel_chip_name} for accelerometer")
        return accel_chip_name

    def get_default_autotune_parameters(
        self,
        steppers: list[str],
    ) -> tuple[int, int, int, int, int, int, int, int]:
        """Return default autotune parameters from current printer config.

        Reads extra_hysteresis from [autotune_tmc stepper_x] and
        tbl/toff/tpfd from the TMC driver settings.

        Args:
            steppers (list[str]): The main and secondary stepper.

        Returns:
            tuple[int, int, int, int, int, int, int, int]: extra_hyst_min, extra_hyst_max,
                tbl_min, tbl_max, toff_min, toff_max, tpfd_min, tpfd_max
        """
        tbl_max = tbl_min = self.driver_settings[steppers[0]].get("driver_tbl")
        toff_max = toff_min = self.driver_settings[steppers[0]].get("driver_toff")
        tpfd_max = tpfd_min = self.driver_settings[steppers[0]].get("driver_tpfd", 0)
        autotune_cfg = self.settings.get(f"autotune_tmc {steppers[0]}", {})
        eh = int(autotune_cfg.get("extra_hysteresis", 0))
        extra_hyst_min = extra_hyst_max = eh
        return (extra_hyst_min, extra_hyst_max, tbl_min, tbl_max, toff_min, toff_max, tpfd_min, tpfd_max)

    def configure_speed_limits(
        self,
        min_speed: None | float,
        max_speed: None | float,
        speed_change_step: None | float,
        measure_time: float,
        axes: list[str],
        steppers: list[str],
        a_axis_min: float,
        a_axis_max: float,
        acceleration: float,
    ) -> tuple[float, float, float]:
        """Configure speed limits.

        Args:
            min_speed (float | str): The in speed value, or can be set to None
                to auto calculate the value over the required RPM value.
            max_speed (float | str): The max speed value, or can be set to None
                to auto calculate the value over the required RPM value.
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
                gear_ratio = ((1, 1),)
            gear_ratio = tuple(float(r) for r in gear_ratio[0])
            full_steps_per_rotation = self.stepper_settings[steppers[0]].get(
                "full_steps_per_rotation", 200
            )

            steps_multiplier = (
                full_steps_per_rotation
                / 200
                / (float(gear_ratio[0]) / float(gear_ratio[1]))
                * rotation_dist
                / 60  # to convert to mm/s from mm/min
            )

            if min_speed is None:
                min_speed = float(self.required_rpm[0] * steps_multiplier)
            else:
                min_speed = float(min_speed)

            if max_speed is None:
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

            if speed_change_step is None:
                speed_change_step = self.required_rpm[2] * steps_multiplier
            else:
                speed_change_step = float(speed_change_step)
        else:
            # Protect not defined speed & converting str -> float
            if min_speed is None or max_speed is None:
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
            travel_distance (None | float): The travel distance, or can be set
                to None to calculate the travel distance with the
                `measure_time`, `max_speed` and `accel_decel_distance`.

        Returns:
            float: The calculated travel distance.
        """
        # Calculate min required toolhead travel distance from speed,
        # acceleration and time
        accel_decel_distance = max_speed**2 / acceleration
        auto_travel_distance = accel_decel_distance + (max_speed * measure_time)
        if self.debug:
            self.gcode.respond_info(
                f"Acceleration & deceleration zone = {accel_decel_distance} mm"
            )
            self.gcode.respond_info(
                "Auto calculated min required travel distance = "
                f"{auto_travel_distance} mm"
            )

        # Protect exceeding axis limits & calculate travel distance
        if travel_distance is None:
            if a_axis_min + auto_travel_distance > a_axis_max:
                raise self.printer.command_error(
                    f"WARNING!!! Required travel distance on axis ({axes[0]}) "
                    f"({auto_travel_distance:.1f} mm) is longer than kinematics "
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
                    self.gcode.respond_info(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        "is longer than kinematics allows, lowering..."
                    )
            elif travel_distance < auto_travel_distance:
                travel_distance = auto_travel_distance
                if a_axis_min + auto_travel_distance > a_axis_max:
                    raise self.printer.command_error(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        f"is less than required ({auto_travel_distance:.1f} mm), "
                        "and longer than kinematics allows, please lower "
                        "speed or increase acceleration"
                    )

        return travel_distance

    def display_process_summary(
        self,
        extra_hyst_min: int,
        extra_hyst_max: int,
        tbl_min: int,
        tbl_max: int,
        toff_min: int,
        toff_max: int,
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
            extra_hyst_min (int): The minimum extra hysteresis value.
            extra_hyst_max (int): The maximum extra hysteresis value.
            tbl_min (int): The minimum TBL value.
            tbl_max (int): The maximum TBL value.
            toff_min (int): The minimum TOFF value.
            toff_max (int): The maximum TOFF value.
            min_speed (float): The minimum speed value.
            max_speed (float): The maximum speed value.
            speed_change_step (float): The speed change step value.
            iterations (int): The number of iterations.
            search_method (SearchMethod): The search method.
            a_axis_min (float): The minimum position of the main axis.
            travel_distance (float): The travel distance value.
        """
        if self.measurement_mode == MeasurementMode.Resonances:
            self.gcode.respond_info(
                f"Final max travel distance = {travel_distance:.1f} mm, "
                f"position min = {a_axis_min:.1f}, "
                f"traveling = {a_axis_min:.1f} --> {travel_distance + a_axis_min:.1f}"
            )
            self.gcode.respond_info(
                f"Start find resonances mode\n"
                f"Method            : {search_method}\n"
                f"speed             : {min_speed:.2f} --> {max_speed:.2f} mm/s with "
                f"{speed_change_step:.2f} mm/s step\n"
                f"EXTRA_HYSTERESIS  : {extra_hyst_min}\n"
                f"TBL               : {tbl_min}\n"
                f"TOFF              : {toff_min}"
            )
        else:
            self.gcode.respond_info(
                f"Final travel distance = {travel_distance:.1f} mm, "
                f"position min = {a_axis_min:.1f}, "
                f"traveling = {a_axis_min:.1f} --> {travel_distance + a_axis_min:.1f}"
            )
            self.gcode.respond_info(
                "Start autotune parameter sweep\n"
                f"Method            : {search_method}\n"
                f"speed             : {min_speed:.2f} --> {max_speed:.2f} mm/s\n"
                f"iterations        : {iterations}\n"
                f"EXTRA_HYSTERESIS  : {extra_hyst_min} --> {extra_hyst_max}\n"
                f"TBL               : {tbl_min} --> {tbl_max}\n"
                f"TOFF              : {toff_min} --> {toff_max}"
            )

    def home(self) -> None:
        """Home."""
        self.gcode.run_script_from_command("G28 X Y Z")
        self.toolhead.wait_moves()

    def get_static_acceleration(self) -> list[Accel_Measurement]:
        """Get static accelerometer samples.

        This will help removing the effect of gravity + static vibrations from
        real vibration data.

        Returns:
            list[Accel_Measurement]: The measurement data samples.
        """
        self.toolhead.wait_moves()
        with AccelerometerMeasure(self.accelerometer) as accelerometer_measurement:
            self.toolhead.dwell(5.0)
        return accelerometer_measurement.samples

    def measure_vibrations(
        self,
        coord_generator: CoordGenerator,
        travel_distance: float,
        speed: float,
    ) -> list[Accel_Measurement]:
        """Perform vibration measurement.

        Args:
            coord_generator (CoordGenerator): The coordinate generator.
            travel_distance (float): The travel distance.
            speed (float): The speed for the measurement.
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            name (str): The name of the measurement.

        Returns:
            list[Accel_Measurement]: The measurement data samples.
        """
        # Start accel_chip data collection
        with AccelerometerMeasure(
            accelerometer=self.accelerometer
        ) as accelerometer_measurement:
            next_coord = coord_generator.next(travel_distance)
            self.toolhead.manual_move(next_coord, speed)
        # Move to the initial position
        self.toolhead.manual_move(self.initial_position, self.travel_speed)
        coord_generator.current_coord.x = self.initial_position.x
        coord_generator.current_coord.y = self.initial_position.y
        coord_generator.current_coord.z = self.initial_position.z
        self.toolhead.wait_moves()
        self.number_of_real_samples += 1
        return accelerometer_measurement.samples

    def calculate_frequency(self, tbl: int, toff: int) -> float:
        """Calculate frequency based on TBL and TOFF values.

        Args:
            tbl (int): The TBL value.
            toff (int): The TOFF value.

        Returns:
            float: The calculated frequency in Hz.
        """
        return 1 / (
            2 * (12 + 32 * toff) * 1 / (1000000 * FCLK)
            + 2 * 1 / (1000000 * FCLK) * 16 * (1.5**tbl)
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
                initial_direction = Coord((-1, 1, 0)).unitize()
            else:
                initial_direction = Coord((0, 1, 0))
        elif axes[0] == "z":
            initial_direction = Coord((0, 0, 1))
        return initial_direction

    def wait_for_file_write(self, data_path: str) -> None:
        """Wait for file write to complete.

        Args:
            data_path (str): The path to the CSV file.
        """
        start_time = self.reactor.monotonic()
        max_wait_time = 10  # seconds
        prev_size = -1
        curr_size = 0

        while True:
            curr_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
            if curr_size > 0 and curr_size == prev_size:
                break

            prev_size = curr_size

            # Yield control back to Klipper for 100ms
            self.reactor.pause(self.reactor.monotonic() + 0.1)

            if self.reactor.monotonic() - start_time > max_wait_time:
                self.gcode.respond_info(f"Timeout waiting for file: {data_path}")
                break

    def calc_static_acceleration_magnitude(
        self, samples: list[Accel_Measurement]
    ) -> tuple[float, float, float]:
        """Calculate static acceleration data from CSV file.

        Args:
            samples (list[Accel_Measurement]): The list of acceleration data
                samples.

        Returns:
            tuple[float, float, float]: Mean static acceleration vector.
        """
        accel_x = np.array([sample.accel_x for sample in samples])
        accel_y = np.array([sample.accel_y for sample in samples])
        accel_z = np.array([sample.accel_z for sample in samples])
        # Return the mean of each axis as the baseline vector
        return (
            float(np.mean(accel_x)),
            float(np.mean(accel_y)),
            float(np.mean(accel_z)),
        )

    def calc_magnitude(
        self, samples: list[Accel_Measurement], static_data: tuple[float, float, float]
    ) -> float:
        """Calculate median magnitude of acceleration data from CSV file.

        Args:
            samples (list[Accel_Measurement]): The list of acceleration data samples.
            static_data (tuple[float, float, float]): Mean static acceleration
                vector.

        Returns:
            float: Median magnitude of acceleration data.
        """
        accel_x = np.array([sample.accel_x for sample in samples]) - static_data[0]
        accel_y = np.array([sample.accel_y for sample in samples]) - static_data[1]
        accel_z = np.array([sample.accel_z for sample in samples]) - static_data[2]

        # calculate the sample rate
        duration = samples[-1].time - samples[0].time
        number_of_samples = len(samples)
        sample_rate = number_of_samples / duration

        magnitudes = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        # Create a 4th order Butterworth filter
        cutoff_freq = 300  # Hz
        nyquist_freq = sample_rate / 2
        normal_cutoff = cutoff_freq / nyquist_freq

        b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)
        filtered_magnitudes = signal.filtfilt(b, a, magnitudes)

        # Percentile Trimming (The "Middle 60%")
        lower_bound = np.percentile(filtered_magnitudes, 20)
        upper_bound = np.percentile(filtered_magnitudes, 80)
        trimmed_magnitudes = filtered_magnitudes[
            (filtered_magnitudes >= lower_bound) & (filtered_magnitudes <= upper_bound)
        ]
        return float(np.median(trimmed_magnitudes))

    def chopper_tune(
        self,
        axis: str,
        extra_hyst_min: int = DEFAULT_REGISTER_VALUES["EXTRA_HYSTERESIS"][0],
        extra_hyst_max: int = DEFAULT_REGISTER_VALUES["EXTRA_HYSTERESIS"][1],
        tbl_min: int = DEFAULT_REGISTER_VALUES["TBL"][0],
        tbl_max: int = DEFAULT_REGISTER_VALUES["TBL"][1],
        toff_min: int = DEFAULT_REGISTER_VALUES["TOFF"][0],
        toff_max: int = DEFAULT_REGISTER_VALUES["TOFF"][1],
        tpfd_min: int = DEFAULT_REGISTER_VALUES["TPFD"][0],
        tpfd_max: int = DEFAULT_REGISTER_VALUES["TPFD"][1],
        min_speed: None | float = None,
        max_speed: None | float = None,
        speed_change_step: None | float = None,
        search_method: SearchMethod = SearchMethod.BruteForce,
        travel_distance: None | float = None,
        direction: int = 1,
        accel_chip_name: str = "default",
        run_plotter: bool = True,
        compare_with: None | str = None,
    ) -> None | dict:
        """Measure vibrations and tune stepper motors for low noise.

        Args:
            axis (str): Axis to tune. Should be one of ["x", "y", "z"].
            extra_hyst_min (int): The min EXTRA_HYSTERESIS value.
            extra_hyst_max (int): The max EXTRA_HYSTERESIS value.
            tbl_min (int): The min TBL value.
            tbl_max (int): The max TBL value.
            toff_min (int): The min TOFF value.
            toff_max (int): The max TOFF value.
            min_speed (None | float): The min speed, or None to auto-calculate.
            max_speed (None | float): The max speed, or None to auto-calculate.
            speed_change_step (None | float): Speed step per iteration.
            search_method (SearchMethod): The search method.
            travel_distance (None | float): Travel distance, or None to
                auto-calculate.
            direction (int): Movement direction: 1 or -1.
            accel_chip_name (str): The name of the acceleration chip.
            run_plotter (bool): If True, generate magnitude graphs after measurement.
            compare_with (None | str): Name of previous sample set to compare with.

        Returns:
            None | dict: The best parameters found, or None if tuning was not
                done using the adaptive method.
        """
        # reset previous run data
        self.reset_sample_data()
        self.reset_registers()

        self.search_method = search_method

        # Force brute_force in vibration measurement mode
        if self.measurement_mode == MeasurementMode.Resonances:
            self.search_method = SearchMethod.BruteForce

        self.gcode.respond_info(f"Selected {self.search_method} as search method")

        measure_time = self.measure_time / 1000
        # Find the steppers count of the main axis
        self.registers["stepper_count"] = self.get_stepper_count(axis)

        self.driver, self.sense_resistor = self.detect_driver(stepper=axis)

        axes, steppers = self.get_axes_and_steppers(axis)

        a_axis_min, a_axis_max, a_axis_mid, b_axis_mid = self.get_axis_limits(axes)
        acceleration, self.travel_speed = self.get_travel_speed_and_acceleration(axes)
        accel_chip_name = self.get_accelerometer_chip(accel_chip_name, axis)
        self.accelerometer = self.printer.lookup_object(accel_chip_name)

        if self.measurement_mode == MeasurementMode.Resonances:
            # In resonance measurement mode, read current autotune config values
            (
                extra_hyst_min,
                extra_hyst_max,
                tbl_min,
                tbl_max,
                toff_min,
                toff_max,
                tpfd_min,
                tpfd_max,
            ) = self.get_default_autotune_parameters(steppers)

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
        self.display_process_summary(
            extra_hyst_min,
            extra_hyst_max,
            tbl_min,
            tbl_max,
            toff_min,
            toff_max,
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

        # try:
        #     self.toolhead.set_max_velocities(None, acceleration, None, None)
        # except AttributeError:
        # This is either an older Klipper version or Kalico
        self.gcode.run_script_from_command(
            f"SET_VELOCITY_LIMIT ACCEL={acceleration}\n"
            f"SET_VELOCITY_LIMIT ACCEL_TO_DECEL={acceleration}"
        )
        self.toolhead.wait_moves()

        # move away from the middle exactly half or a travel distance
        self.initial_position -= self.initial_direction * (travel_distance / 2)
        self.toolhead.manual_move(self.initial_position, self.travel_speed)
        self.toolhead.wait_moves()

        # Measure accelerometer noise
        samples = self.get_static_acceleration()
        self.static_acceleration_vector = self.calc_static_acceleration_magnitude(
            samples
        )
        self.static_acceleration_magnitude = float(
            np.linalg.norm(self.static_acceleration_vector)
        )

        # set the global start time here
        self.global_start_time = self.reactor.monotonic()

        nv = self.static_acceleration_vector
        self.gcode.respond_info(
            f"Static acceleration vector    = {nv[0]:0.1f} {nv[1]:0.1f} {nv[2]:0.1f} mm/s²\n"
            f"Static acceleration magnitude = {self.static_acceleration_magnitude:.1f} mm/s²\n"
            "(HINT: this should be close to earth's gravity of 9806 mm/s²)"
        )

        # Create the coordinate generator
        coord_generator = CoordGenerator(
            direction=self.initial_direction,
            start_coord=self.initial_position,
        )

        # calculated vars
        self.travel_distance = travel_distance
        self.coord_generator = coord_generator
        self.accel_chip_name = accel_chip_name
        self.steppers = steppers

        # bounds
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.speed_change_step = speed_change_step
        self.extra_hyst_min = extra_hyst_min
        self.extra_hyst_max = extra_hyst_max

        self.tbl_min = tbl_min
        self.tbl_max = tbl_max
        self.toff_min = toff_min
        self.toff_max = toff_max

        self.bounds = [
            (self.extra_hyst_min, self.extra_hyst_max),
            (self.tbl_min, self.tbl_max),
            (self.toff_min, self.toff_max),
        ]
        best_parameters = self.search_best_parameters()
        if self.measurement_mode == MeasurementMode.Resonances:
            max_vibrations_and_speed = sorted(
                self.speed_vs_vibrations, key=lambda x: x[1]
            )[-1]
            self.gcode.respond_info(
                "Finding resonances took "
                f"{self.convert_seconds_to_hms(self.get_time_elapsed())}\n"
                "Max vibrations seems to be at "
                f"{max_vibrations_and_speed[0]:0.1f} mm/s"
            )

        self.toolhead.dwell(0.5)
        self.toolhead.manual_move((a_axis_mid,), self.travel_speed)
        self.toolhead.wait_moves()
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_plotter:
            self.gcode.respond_info("Magnitude graphs generation...")
            self.gcode.respond_info("This may take a while, please wait")
            self.plot_data(date_stamp=now)

        # Store data
        self.store_data(date_stamp=now)

        if self.measurement_mode != MeasurementMode.Resonances:
            # Save Config
            self.save_configs(best_parameters)

            # Compare best result with previous samples
            if best_parameters and compare_with is not None:
                sample_name = self.generate_sample_name(
                    best_parameters["extra_hysteresis"],
                    best_parameters["tbl"],
                    best_parameters["toff"],
                    best_parameters["speed"],
                )
                self.compare_results(
                    sample_name,
                    self.samples[sample_name],
                    compare_with,
                )

        # reset samples related data
        self.total_expected_samples = -1
        self.number_of_samples = 0
        self.number_of_real_samples = 0
        self.samples = {}

        return best_parameters

    def generate_sample_name(
        self,
        extra_hysteresis: int,
        tbl: int,
        toff: int,
        tpfd: int,
        speed: float,
    ) -> str:
        """Generate a sample name based on the autotune parameters.

        Args:
            extra_hysteresis (int): The extra hysteresis value.
            tbl (int): The TBL value.
            toff (int): The TOFF value.
            tpfd (int): The TPFD value.
            speed (float): The speed value.

        Returns:
            str: The generated sample name.
        """
        return (
            f"eh={extra_hysteresis}_"
            f"tbl={tbl}_"
            f"toff={toff}_"
            f"tpfd={tpfd}_"
            f"speed={speed:.2f}"
        )

    def compare_results(
        self,
        sample_name: str,
        sample_result: float,
        previous_samples_name: str,
    ) -> tuple[None | float, None | float]:
        """Compare current sample result with previous samples.

        Args:
            sample_name (str): The name of the current sample.
            sample_result (float): The result of the current sample.
            previous_samples_name (str): The name of the previous samples file.

        Returns:
            tuple[None | float, None | float]: The percentile of the given
                value within the previous samples, and the previously measured
                value for the sample name.
        """
        previous_sample_path = os.path.join(
            RESULTS_FOLDER, f"{previous_samples_name}.json"
        )
        if not os.path.exists(previous_sample_path):
            return None, None

        with open(previous_sample_path) as f:
            previous_samples = json.load(f)

        previous_values = np.array(list(previous_samples.values()))
        previous_sample_result = previous_samples.get(sample_name)
        percentile = float((previous_values <= sample_result).mean())

        # report comparison results
        message = (
            "Comparison with previous samples:\n---------------------------------\n"
        )
        if percentile is not None:
            message += f"Sample    : {sample_name}\nPercentile: {percentile:.2%}\n"
        if previous_sample_result is not None:
            message += (
                f"Previous value\n"
                f"for the same\n"
                f"sample    : {previous_sample_result:.1f} mm/s²"
            )
        if percentile is None and previous_sample_result is None:
            message += "No previous sample data found for comparison!!!\n"
        self.gcode.respond_info(message)
        return percentile, previous_sample_result

    def get_time_elapsed(self) -> float:
        """Get elapsed time since the start of the process.

        Returns:
            float: The time elapsed in seconds.
        """
        return self.reactor.monotonic() - self.global_start_time

    def get_remaining_time(self) -> float:
        """Estimate remaining time for the process.

        Returns:
            float: The estimated remaining time in seconds.
        """
        time_passed = self.get_time_elapsed()
        total_estimated_process_time = (
            time_passed / self.number_of_samples * self.total_expected_samples
        )
        return max(total_estimated_process_time - time_passed, 0)

    def convert_seconds_to_hms(self, seconds: float) -> str:
        """Convert seconds to hours, minutes, and seconds.

        Args:
            seconds (float): The time in seconds.

        Returns:
            str: The time in "HHh MMm SSs" format.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        # do not include hours or minutes if zero
        hours_str = f"{hours}h" if hours > 0 else ""
        minutes_str = f"{minutes}m" if minutes > 0 or hours > 0 else ""

        return f"{hours_str}{minutes_str}{secs}s"

    def progress_report(
        self, measured_vibrations: float = 0.0, iteration: int = 0, cached: bool = False
    ) -> None:
        """Report progress.

        Args:
            measured_vibrations (float): The measured vibrations.
            iteration (int): The current iteration. Only report progress on the
                first iteration.
            cached (bool): Whether the vibrations value is from a cached sample.
        """
        if iteration == 0:
            if self.total_expected_samples > 0:
                percent_complete = (
                    self.number_of_samples / self.total_expected_samples
                ) * 100
                self.gcode.respond_info(
                    f"Sample {'(cached)' if cached else '        '}: "
                    f"{self.number_of_samples}/"
                    f"{self.total_expected_samples} "
                    f"({percent_complete:0.1f}%) | "
                    f"E: {self.convert_seconds_to_hms(self.get_time_elapsed())} | "
                    f"R: {self.convert_seconds_to_hms(self.get_remaining_time())}"
                )
            else:
                self.gcode.respond_info(
                    f"Sample         : {self.number_of_samples} | "
                    f"E: {self.convert_seconds_to_hms(self.get_time_elapsed())}"
                )
        self.gcode.respond_info(
            f"Iteration {iteration + 1:<5d}: {measured_vibrations:0.1f} mm/s²"
        )

    def objective_function(self, params: list[float]) -> float:
        """Objective function for optimization.

        Args:
            params (list[float]): The parameters to optimize.
                Order: [extra_hysteresis, tbl, toff, tpfd, speed*100]

        Returns:
            float: The average measured vibrations.
        """
        self.gcode.respond_info("-------------------------------")
        extra_hysteresis, tbl, toff, tpfd, speed = [round(p) for p in params]

        # convert speed back to the correct range
        speed = float(speed) / 100

        self.gcode.respond_info(
            f"extra_hysteresis={extra_hysteresis} "
            f"tbl={tbl} "
            f"toff={toff} "
            f"tpfd={tpfd} "
            f"speed={speed:.2f}"
        )

        # Apply autotune parameters
        self.apply_autotune(self.steppers, extra_hysteresis, tbl, toff, tpfd)
        sample_name = self.generate_sample_name(
            extra_hysteresis=extra_hysteresis,
            tbl=tbl,
            toff=toff,
            tpfd=tpfd,
            speed=speed,
        )
        if sample_name in self.samples:
            cached_vibrations = self.samples[sample_name]
            self.number_of_samples += 1  # consider this as a sample
            self.progress_report(cached_vibrations, cached=True)
            return cached_vibrations

        total_vibrations = 0
        self.number_of_samples += 1
        travel_distance = self.travel_distance
        if self.measurement_mode == MeasurementMode.Resonances:
            travel_distance = self.travel_distance * (speed / self.max_speed)

        for iteration in range(self.iterations):
            samples = self.measure_vibrations(
                self.coord_generator,
                travel_distance,
                speed,
            )
            measured_vibrations = self.calc_magnitude(
                samples=samples,
                static_data=self.static_acceleration_vector,
            )

            total_vibrations += measured_vibrations
            self.progress_report(measured_vibrations, iteration)

            # Allow other Klipper tasks to run between iterations
            self.reactor.pause(self.reactor.monotonic() + 0.05)

        total_vibrations /= self.iterations
        if self.iterations > 1:
            self.gcode.respond_info(f"Mean vibrations: {total_vibrations:0.1f} mm/s²")
        # store best result for the last report
        self.best_result = min(self.best_result, total_vibrations)

        self.samples[sample_name] = total_vibrations
        self.speed_vs_vibrations.append((speed, total_vibrations))

        # Allow other Klipper tasks to run between iterations
        self.reactor.pause(self.reactor.monotonic() + 0.05)

        return total_vibrations

    def progressive_search_loop(
        self,
        param_min: int,
        param_max: int,
        param_index: int,
        best_params: list[int],
    ) -> list[int]:
        """Progressive search loop for a single parameter.

        Args:
            param_min (int): The minimum value of the parameter.
            param_max (int): The maximum value of the parameter.
            param_index (int): The index of the parameter in the best_params list.
            best_params (list[int]): The current best parameters.
        """
        best_mag = float("inf")
        for param in range(param_min, param_max + 1):
            best_params[param_index] = param
            mag = self.objective_function(best_params)
            if mag < best_mag:
                best_mag = mag
                best_param = param
        best_params[param_index] = best_param
        return best_params

    def perform_brute_force_search(self) -> list[int]:
        """Perform brute-force search for optimal autotune parameters.

        Returns:
            list[int]: The best parameters found.
        """
        bounds = [
            slice(self.extra_hyst_min, self.extra_hyst_max + 1, 1),
            slice(self.tbl_min, self.tbl_max + 1, 1),
            slice(self.toff_min, self.toff_max + 1, 1),
            slice(
                int(self.min_speed * 100),
                int(self.max_speed * 100) + 1,
                int(self.speed_change_step * 100),
            ),
        ]

        # update total expected samples
        total_eh_steps = self.extra_hyst_max - self.extra_hyst_min + 1
        total_tbl_steps = self.tbl_max - self.tbl_min + 1
        total_toff_steps = self.toff_max - self.toff_min + 1
        total_speed_steps = (
            int(self.max_speed * 100) - int(self.min_speed * 100)
        ) // int(self.speed_change_step * 100) + 1
        self.total_expected_samples = (
            total_eh_steps
            * total_tbl_steps
            * total_toff_steps
            * total_speed_steps
        )

        result = brute(
            self.objective_function,
            bounds,
            finish=None,  # disable local optimization at the end
        )

        return [round(p) for p in result]

    def perform_adaptive_search(self) -> list[int]:
        """Perform adaptive search for optimal parameters.

        Returns:
            list[int]: The best parameters found.
        """
        # Adaptive search
        # 'strategy' and 'popsize' are tuned to reduce total measurements
        # 'tol' can be higher since our parameters are discrete

        bounds = [
            (self.extra_hyst_min, self.extra_hyst_max),
            (self.tbl_min, self.tbl_max),
            (self.toff_min, self.toff_max),
            (self.min_speed * 100, self.max_speed * 100),
        ]

        number_of_changing_params = 0
        for bound in bounds:
            if bound[0] != bound[1]:
                number_of_changing_params += 1
        number_of_changing_params += 1  # add one for safety

        maxiter = 10
        popsize = 5
        self.total_expected_samples = maxiter * popsize * number_of_changing_params
        result = differential_evolution(
            self.objective_function,
            bounds,
            init="sobol",
            strategy="best1bin",
            maxiter=maxiter,
            popsize=popsize,  # Total evaluations = maxiter * popsize * N_params
            tol=0.05,  # Higher tolerance for discrete parameters
            mutation=(0.3, 0.8),
            recombination=0.9,  # Increased for faster parameter mixing
            polish=False,  # Polish uses local minimize, which we avoid for discrete
            updating="immediate",  # Uses best results immediately
        )

        return [round(p) for p in result.x]

    def perform_progressive_search(self) -> list[int]:
        """Perform progressive search for optimal parameters.

        Returns:
            list[int]: The best parameters found.
        """
        self.gcode.respond_info(
            "Starting Progressive (Trinamic Flowchart) Optimization..."
        )

        # params order: [extra_hysteresis, tbl, toff, tpfd, speed*100]
        best_toff = self.toff_min if self.toff_min > 0 else 3
        best_tbl = 2
        best_eh = self.extra_hyst_min
        best_tpfd = self.tpfd_min
        best_params = [
            best_eh,
            best_tbl,
            best_toff,
            best_tpfd,
            int(self.min_speed * 100),
        ]
        self.total_expected_samples = (
            (self.toff_max - self.toff_min + 1)
            + (self.tbl_max - self.tbl_min + 1)
            + (self.extra_hyst_max - self.extra_hyst_min + 1)
            + (self.tpfd_max - self.tpfd_min + 1)
        )

        self.gcode.respond_info("Step 1: Optimizing TOFF...")
        best_params = self.progressive_search_loop(
            param_min=self.toff_min,
            param_max=self.toff_max,
            param_index=2,
            best_params=best_params,
        )
        self.gcode.respond_info(f"-> Best TOFF found: {best_params[2]}")

        self.gcode.respond_info("Step 2: Optimizing TBL...")
        best_params = self.progressive_search_loop(
            param_min=self.tbl_min,
            param_max=self.tbl_max,
            param_index=1,
            best_params=best_params,
        )
        self.gcode.respond_info(f"-> Best TBL found: {best_params[1]}")

        self.gcode.respond_info("Step 3: Optimizing EXTRA_HYSTERESIS...")
        best_params = self.progressive_search_loop(
            param_min=self.extra_hyst_min,
            param_max=self.extra_hyst_max,
            param_index=0,
            best_params=best_params,
        )
        self.gcode.respond_info(f"-> Best EXTRA_HYSTERESIS found: {best_params[0]}")

        self.gcode.respond_info("Step 4: Optimizing TPFD...")
        best_params = self.progressive_search_loop(
            param_min=self.tpfd_min,
            param_max=self.tpfd_max,
            param_index=3,
            best_params=best_params,
        )
        self.gcode.respond_info(f"-> Best TPFD found: {best_params[3]}")

        return best_params

    def search_best_parameters(self) -> list[int]:
        """Run the parameter search process.

        This can use either brute-force or adaptive optimization methods depending
        on the selected search method.

        Returns:
            dict: The best parameters found.
        """
        self.gcode.respond_info("Starting optimization...")
        start_time = self.reactor.monotonic()

        if self.measurement_mode == MeasurementMode.Resonances:
            self.gcode.respond_info(
                f"eh={self.extra_hyst_min} "
                f"tbl={self.tbl_min} "
                f"toff={self.toff_min} "
                f"speed={self.min_speed:.2f} --> {self.max_speed:.2f}"
            )

        # set initial autotune parameters
        self.apply_autotune(self.steppers, self.extra_hyst_min, self.tbl_min, self.toff_min, self.tpfd_min)

        if self.search_method == SearchMethod.BruteForce:
            best_params = self.perform_brute_force_search()
        elif self.search_method == SearchMethod.Adaptive:
            best_params = self.perform_adaptive_search()
        elif self.search_method == SearchMethod.Progressive:
            best_params = self.perform_progressive_search()

        # update overall best params with final results
        overall_best_params = {
            "extra_hysteresis": best_params[0],
            "tbl": best_params[1],
            "toff": best_params[2],
            "tpfd": best_params[3],
            "speed": float(best_params[4]) / 100,
        }

        duration = self.reactor.monotonic() - start_time

        result_message = (
            f"Optimization Completed in {self.convert_seconds_to_hms(duration)}\n"
            f"Number of samples      : {self.number_of_samples}\n"
            f"Best Score             : {self.best_result:.1f} mm/s²\n\n"
            "Best Autotune Parameters\n"
            "------------------------\n"
            f"speed                  : {overall_best_params['speed']:.2f} mm/s\n"
            f"extra_hysteresis       : {overall_best_params['extra_hysteresis']}\n"
            f"driver_TBL             : {overall_best_params['tbl']}\n"
            f"driver_TOFF            : {overall_best_params['toff']}\n"
            f"driver_TPFD            : {overall_best_params['tpfd']}"
        )
        self.gcode.respond_info(result_message)

        # Apply best parameters
        self.apply_autotune(
            self.steppers,
            overall_best_params["extra_hysteresis"],
            overall_best_params["tbl"],
            overall_best_params["toff"],
            overall_best_params["tpfd"],
        )

        return overall_best_params

    def save_configs(self, best_parameters: dict | None) -> None:
        """Save the best autotune parameters to printer.cfg.

        Args:
            best_parameters (dict | None): The best parameters found.
        """
        if best_parameters is None:
            return

        for stepper_index in range(self.registers["stepper_count"]):
            suffix = str(stepper_index) if stepper_index > 0 else ""
            section = f"autotune_tmc {self.steppers[0]}{suffix}"
            if "extra_hysteresis" in best_parameters:
                self.configfile.set(
                    section, "extra_hysteresis",
                    str(best_parameters["extra_hysteresis"]),
                )
            if "tbl" in best_parameters:
                self.configfile.set(
                    section, "driver_TBL", str(best_parameters["tbl"])
                )
            if "toff" in best_parameters:
                self.configfile.set(
                    section, "driver_TOFF", str(best_parameters["toff"])
                )
            if "tpfd" in best_parameters:
                self.configfile.set(
                    section, "driver_TPFD", str(best_parameters["tpfd"])
                )

        self.gcode.respond_info(
            "Best autotune parameters saved to printer.cfg, run SAVE_CONFIG to apply."
        )

    def plot_data(self, date_stamp: None | str = None) -> None:
        """Plot the collected vibration data.

        Args:
            date_stamp (None | str): The timestamp string to use in filenames.
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        if date_stamp is None:
            date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        params = [
            reversed(list(self.samples.items())),
            sorted(self.samples.items(), key=lambda x: x[1]),
        ]
        names = ["as_measured", "sorted"]
        plot_html_paths = []
        for param, name in zip(params, names):
            fig = go.Figure()
            for entry in param:
                toff = int(entry[0].split("_")[2].split("=")[1])
                color = COLORS[toff % len(COLORS)]
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
                xaxis_title="Median Magnitude (mm/s²)",
                yaxis_title="Parameters",
                coloraxis_showscale=True,
            )
            plot_html_path = os.path.join(
                RESULTS_FOLDER,
                f"{date_stamp}"
                f"_autotune_sweep_{name}"
                f"_{self.accel_chip_name}"
                f"_{self.steppers[0]}"
                ".html",
            )
            plot_html_paths.append(plot_html_path)
            # check if the RESULTS_FOLDER exists before writing
            if not os.path.exists(RESULTS_FOLDER):
                os.makedirs(RESULTS_FOLDER)

            pio.write_html(fig, plot_html_path, auto_open=False)

        # Export Info
        self.gcode.respond_info("Access to interactive plot at:")
        for plot_html_path in plot_html_paths:
            self.gcode.respond_info(f"{plot_html_path}")

    def store_data(self, date_stamp: None | str = None) -> None:
        """Store collected sample data to a JSON file.

        Args:
            date_stamp (None | str): The timestamp string to use in filenames.
        """
        if date_stamp is None:
            date_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(
            RESULTS_FOLDER,
            f"{date_stamp}"
            f"_autotune_sweep_data"
            f"_{self.accel_chip_name}"
            f"_{self.steppers[0]}"
            ".json",
        )
        with open(json_path, mode="w") as file:
            json.dump(self.samples, file, indent=4)
        self.gcode.respond_info(f"Sample data saved to: {json_path}")

    def cmd_chopper_tune(self, gcmd: GCodeCommand) -> bool:
        """Tune stepper values.

        Args:
            gcmd (GCodeCommand): The G-Code command.

        Returns:
            bool: True if command completed successfully, False otherwise.
        """
        self.reactor.register_callback(
            lambda e: self.parse_args_and_run_optimization(gcmd)
        )
        return True

    def parse_args_and_run_optimization(self, gcmd: GCodeCommand) -> None:
        """Collect data from G-Code command and run optimization.

        Args:
            gcmd (GCodeCommand): The G-Code command.
        """
        try:
            axis = gcmd.get("AXIS", "x").lower()
            direction = gcmd.get_int("DIRECTION", 1)
            # search_method can be brute_force, adaptive or progressive
            search_method = SearchMethod.to_method(
                gcmd.get("SEARCH_METHOD", "progressive").lower()
            )
            extra_hyst_min = gcmd.get_int(
                "EXTRA_HYSTERESIS_MIN", DEFAULT_REGISTER_VALUES["EXTRA_HYSTERESIS"][0]
            )
            extra_hyst_max = gcmd.get_int(
                "EXTRA_HYSTERESIS_MAX", DEFAULT_REGISTER_VALUES["EXTRA_HYSTERESIS"][1]
            )
            tbl_min = gcmd.get_int("TBL_MIN", DEFAULT_REGISTER_VALUES["TBL"][0])
            tbl_max = gcmd.get_int("TBL_MAX", DEFAULT_REGISTER_VALUES["TBL"][1])
            toff_min = gcmd.get_int("TOFF_MIN", DEFAULT_REGISTER_VALUES["TOFF"][0])
            toff_max = gcmd.get_int("TOFF_MAX", DEFAULT_REGISTER_VALUES["TOFF"][1])
            tpfd_min = gcmd.get_int("TPFD_MIN", DEFAULT_REGISTER_VALUES["TPFD"][0])
            tpfd_max = gcmd.get_int("TPFD_MAX", DEFAULT_REGISTER_VALUES["TPFD"][1])
            min_speed = gcmd.get_float("MIN_SPEED", None)
            max_speed = gcmd.get_float("MAX_SPEED", None)
            speed_change_step = gcmd.get_float("SPEED_CHANGE_STEP", None)
            self.iterations = gcmd.get_int("ITERATIONS", 1)
            travel_distance = gcmd.get_float("TRAVEL_DISTANCE", None)
            accel_chip_name = gcmd.get("ACCELEROMETER", "default").lower()
            compare_with = gcmd.get("COMPARE_WITH", None)

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
                extra_hyst_min=extra_hyst_min,
                extra_hyst_max=extra_hyst_max,
                tbl_min=tbl_min,
                tbl_max=tbl_max,
                toff_min=toff_min,
                toff_max=toff_max,
                tpfd_min=tpfd_min,
                tpfd_max=tpfd_max,
                min_speed=min_speed,
                max_speed=max_speed,
                speed_change_step=speed_change_step,
                search_method=search_method,
                travel_distance=travel_distance,
                direction=direction,
                accel_chip_name=accel_chip_name,
                run_plotter=run_plotter,
                compare_with=compare_with,
            )
        except Exception as e:
            self.gcode.respond_info(traceback.format_exc())

    def cmd_chopper_tune_debug(self, gcmd: GCodeCommand) -> bool:
        """Development debug tool.

        Args:
            gcmd (GCodeCommand): The G-Code command.

        Returns:
            bool: True if command completed successfully, False otherwise.
        """
        try:
            self.gcode.respond_info(f"x stepper count: {self.get_stepper_count('x')}")
            self.gcode.respond_info(f"y stepper count: {self.get_stepper_count('y')}")
            self.gcode.respond_info(f"z stepper count: {self.get_stepper_count('z')}")
        except Exception as e:
            self.gcode.respond_info(traceback.format_exc(e))


def load_config(config: ConfigWrapper) -> ChopperTune:
    """Load the ChopperTune config prefix.

    Args:
        config (ConfigWrapper): The config wrapper.

    Returns:
        ChopperTune: The ChopperTune instance.
    """
    return ChopperTune(config)


def load_config_prefix(config: ConfigWrapper) -> ChopperTune:
    """Load the ChopperTune config prefix.

    Args:
        config (ConfigWrapper): The config wrapper.

    Returns:
        ChopperTune: The ChopperTune instance.
    """
    return ChopperTune(config)
