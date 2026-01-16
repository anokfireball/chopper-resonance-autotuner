"""Chopper Tune extension for Klipper.

TMC drivers registers calibration tool.

Copyright (C) 2024  Alexander Fedorov <altzbox@gmail.com>
Copyright (C) 2024  Maksim Bolgov <maksim8024@gmail.com>

This file may be distributed under the terms of the GNU GPLv3 license.
"""

# Standard Library Imports
from __future__ import annotations

import glob
import operator
import os
import re
import shutil
import time
import traceback
from functools import reduce, wraps
from typing import TYPE_CHECKING, Callable

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
        return f"/tmp/{self.full_name}"  # Klipper saves the measurement files in /tmp/ # noqa: S108

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
        if os.path.exists(self.full_path):
            shutil.move(self.full_path, destination)
        else:
            self.gcode.respond_info(f"File doesn't exist: {self.full_path}")
        return destination


class Coord(list):
    """Custom "list" class for coordinates - add easy access to x, y, z components.

    The difference between the gcode.Coord is that, this class allows attribute
    setting.

    Args:
        t: A list or tuple.
    """
    __slots__ = ()
    def __new__(cls, t):
        if len(t) < 4:
            t = list(tuple(t) + (0,) * (3 - len(t)))
        return list.__new__(cls, t)

    @property
    def x(self) -> float:
        return self[0]

    @x.setter
    def x(self, x) -> None:
        self[0] = x

    @property
    def y(self) -> float:
        return self[1]

    @y.setter
    def y(self, y) -> None:
        self[1] = y

    @property
    def z(self) -> float:
        return self[2]

    @z.setter
    def z(self, z) -> None:
        self[2] = z

    def length(self) -> float:
        """Return the vector length."""
        return float(reduce(lambda x, y: x + y**2, [0, *self])**0.5)

    def unitize(self) -> Self:
        """Make self unit vector."""
        other = self / self.length()
        for i in range(len(self)):
            self[i] = other[i]
        return self

    def __add__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x + other[0], self.y + other[1], self.z + other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i + other for i in self])
        else:
            return super().__add__(other)

    def __iadd__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x + other[0], self.y + other[1], self.z + other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i + other for i in self])
        else:
            return super().__iadd__(other)

    def __sub__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x - other[0], self.y - other[1], self.z - other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i - other for i in self])
        else:
            return super().__add__(other)

    def __mul__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x * other[0], self.y * other[1], self.z * other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i * other for i in self])
        else:
            return super().__mul__(other)

    def __imul__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x * other[0], self.y * other[1], self.z * other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i * other for i in self])
        else:
            return super().__imul__(other)

    def __truediv__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x / other[0], self.y / other[1], self.z / other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i / other for i in self])
        else:
            return super().__truediv__(other)

    def __itruediv__(self, other):
        if isinstance(other, (Coord, list, tuple)):
            return Coord((self.x / other[0], self.y / other[1], self.z / other[2]))
        elif isinstance(other, (int, float)):
            return Coord([i / other for i in self])
        else:
            return super().__itruediv__(other)


class CoordGenerator:
    """A class to generate coordinates/positions for chopper tuning."""

    def __init__(
        self,
        axes: tuple[str, str],
        kinematics: str,
        start_coord: None | Coord = None
    ) -> None:
        self.axes = axes
        self.kinematics = kinematics
        self.direction = self.get_initial_direction()
        if start_coord is None:
            start_coord = Coord((0, 0, 0))
        self.current_coord = start_coord

    def get_initial_direction(self) -> Coord:
        """Return the initial direction based on the axes and kinematics."""
        initial_direction = Coord((0, 0, 0))
        if self.axes[0] == "x":
            if self.kinematics == "corexy":
                initial_direction = Coord((1, 1, 0)).unitize()
            else:
                initial_direction = Coord((1, 0, 0))
        elif self.axes[0] == "y":
            if self.kinematics == "corexy":
                initial_direction = Coord((1, -1, 0)).unitize()
            else:
                initial_direction = Coord((0, 1, 0))
        elif self.axes[0] == "z":
            initial_direction = Coord((0, 0, 1))
        return initial_direction

    def switch_direction(self):
        """Switch direction."""
        self.direction *= (-1, -1, -1)

    def next_position(self, travel_distance: float) -> float:
        """Get the next position.

        Args:
            travel_distance (float): The travel distance.

        Returns:
            float: The next position.
        """
        self.current_coord += (self.direction * travel_distance)
        self.switch_direction()
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

    def detect_driver(self, stepper) -> None | str:
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

    def get_axes_and_steppers(self, axis):
        """Get main and secondary axis / stepper.

        Args:
            axis (str): The to be tuned.

        Returns:
            tuple[list[str], list[str]]: A tuple containing:
                - axes (list[str]): The main and secondary axis.
                - steppers (list[str]): The main and secondary stepper.
        """
        if axis not in ["x", "y", "z"]:
            raise self.printer.command_error(f"WARNING!!! Incorrect axis: {axis}")

        if self.kinematics not in ["corexy", "cartesian"]:
            raise self.printer.command_error(
                f"WARNING!!! Unsupported kinematics: {self.kinematics}"
            )

        if self.kinematics == "corexy":
            if axis in ["x", "y"]:
                if axis == "x":
                    axes = ["x", "y"]
                elif axis == "y":
                    axes = ["y", "x"]
                steppers = ["stepper_x", "stepper_y"]
            elif axis == "z":
                axes = ["z", "x"]
                steppers = ["stepper_z"]
        elif self.kinematics == "cartesian":
            if axis == "x":
                axes = ["x", "y"]
                steppers = ["stepper_x"]
            elif axis == "y":
                axes = ["y", "x"]
                steppers = ["stepper_y"]
            elif axis == "z":
                axes = ["z", "x"]
                steppers = ["stepper_z"]

        return axes, steppers

    def determine_axis_configuration(self, axes):
        """Select main and secondary axis / stepper.

        Args:
            axes (str): The to be tuned.

        Returns:
            tuple[list[str], list[str], float, float, float, int, int]: A tuple
                containing:
                - axes (list[str]): The main and secondary axis.
                - steppers (list[str]): The main and secondary stepper.
                - min_a_axis (float): The minimum position of the main axis.
                - max_a_axis (float): The maximum position of the main axis.
                - mid_a_axis (float): The middle position of the main axis.
                - mid_b_axis (float): The middle position of the secondary axis.
                - acceleration (int): The acceleration for the movement.
                - travel_speed (int): The travel speed for idle movements.
        """
        if axes[0] == "z":
            min_a_axis = (
                max(self.stepper_settings[f"stepper_{axes[0]}"]["position_min"], 0)
                + self.inset
            )
            acceleration = self.settings["printer"]["max_z_accel"]
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_z_velocity", 0) / 2 * 60
        else:
            min_a_axis = (
                self.stepper_settings[f"stepper_{axes[0]}"]["position_min"] + self.inset
            )
            acceleration = self.settings["printer"].get("max_accel")
            # Idle movements speed
            travel_speed = self.settings["printer"].get("max_velocity") / 2 * 60

        max_a_axis = (
            self.stepper_settings[f"stepper_{axes[0]}"]["position_max"] - self.inset
        )
        mid_a_axis = self.stepper_settings[f"stepper_{axes[0]}"]["position_min"] + (
            (
                self.stepper_settings[f"stepper_{axes[0]}"]["position_max"]
                - self.stepper_settings[f"stepper_{axes[0]}"]["position_min"]
            )
            / 2
        )
        mid_b_axis = self.stepper_settings[f"stepper_{axes[1]}"]["position_min"] + (
            (
                self.stepper_settings[f"stepper_{axes[1]}"]["position_max"]
                - self.stepper_settings[f"stepper_{axes[1]}"]["position_min"]
            )
            / 2
        )
        return (
            min_a_axis,
            max_a_axis,
            mid_a_axis,
            mid_b_axis,
            acceleration,
            travel_speed,
        )

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
        if tpfd_min == -1 or tpfd_max != -1:
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
    ):
        """Apply registers.

        Args:
            field (str): The name of the field to set the value of.
            value (int): The value to set to.
            steppers (list[str]): The name of the steppers to set the register
                field values of.
        """
        if field is None or value is None:
            return

        for stepper in steppers:
            for stepper_index in range(self.registers["stepper_count"]):
                # stepper_x,
                # stepper_y,
                # stepper_z, stepper_z1, stepper_z2, stepper_z3, ...
                # don't add index for the first stepper
                stepper_index = str(stepper_index) if stepper_index > 0 else ""
                if self.debug:
                    self.respond_info(
                        f"Setting {field.lower()} "
                        f"from {self.registers[field]} to {value} on {stepper}{stepper_index}"
                    )

                if field.lower() == "curr":
                    self.gcode.run_script_from_command(
                        f"SET_TMC_CURRENT STEPPER={stepper} CURRENT={value / 1000}"
                    )
                else:
                    self.gcode.run_script_from_command(
                        f"SET_TMC_FIELD STEPPER={stepper}{stepper_index} FIELD={field} VALUE={value}"
                    )
        # store the last applied value
        self.registers[field.lower()] = value

    def get_stepper_count(self, axis):
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

    def get_accelerometer_chip(self, accel_chip):
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
        find_resonances: bool,
        current_min: int | str,
        current_max: int | str,
        steppers,
    ) -> tuple[int, int]:
        """Get run current.

        Args:
            find_resonances (bool): Sets the mode to resonance measurement.
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

        if find_resonances:
            current_max = current_min

        return current_min, current_max

    def get_default_stepper_parameters(
        self,
        steppers,
    ):
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
        min_speed,
        max_speed,
        speed_change_step,
        find_resonances,
        measure_time,
        axes,
        steppers,
        min_a_axis,
        max_a_axis,
        acceleration,
    ):
        """Configure speed limits.

        Args:
            min_speed (int | str): The in speed value, or can be set to
                "default" to auto calculate the value over the required RPM
                value.
            max_speed (int | str): The max speed value, or can be set to
                "default" to auto calculate the value over the required RPM
                value.
            speed_change_step (int | str): The step in each iteration the speed
                will be increased to.
            find_resonances (bool): Sets the mode to resonance measurement
                mode.
            measure_time (float): The measurement time in seconds.
            axes (list[str]): The main and secondary axis.
            steppers (list[str]): The main and secondary stepper.
            min_a_axis (float): The minimum position of the main axis.
            max_a_axis (float): The maximum position of the main axis.
            acceleration (float): The acceleration for the movement.
        """
        # In vibration measurement mode, search and takes registers from printer.cfg, set speed range
        if find_resonances:
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
                                + 4 * acceleration * (max_a_axis - min_a_axis)
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
        axes,
        min_a_axis,
        max_a_axis,
        max_speed,
        acceleration,
        measure_time,
        travel_distance,
    ) -> float:
        """Calculate travel distance.

        Args:
            axes (list[str]): The main and secondary axis.
            min_a_axis (float): The minimum position of the main axis.
            max_a_axis (float): The maximum position of the main axis.
            max_speed (float): The maximum speed of the main axis.
            acceleration (float): The acceleration of the main axis.
            measure_time (float): The measurement time in seconds.
            travel_distance (int | str): The travel distance, or can be set to
                "default" to calculate the travel distance with the
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
                f"Auto calculated min required travel distance = {auto_travel_distance} mm"
            )

        # Protect exceeding axis limits & calculate travel distance
        if travel_distance == "default":
            if min_a_axis + auto_travel_distance > max_a_axis:
                raise self.printer.command_error(
                    f"WARNING!!! Required travel distance on axis ({axes[0]}) "
                    f"({auto_travel_distance:.2f} mm) is longer than kinematics "
                    "allows, please lower speed or increase acceleration"
                )

            travel_distance = auto_travel_distance
        else:
            travel_distance = int(travel_distance)
            if min_a_axis + travel_distance > max_a_axis:
                travel_distance = max_a_axis - min_a_axis
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
                if min_a_axis + auto_travel_distance > max_a_axis:
                    raise self.printer.command_error(
                        f"WARNING!!! Travel distance on axis ({axes[0]}) "
                        f"is less than required ({auto_travel_distance:.2f} mm), "
                        "and longer than kinematics allows, please lower "
                        "speed or increase acceleration"
                    )

        return travel_distance

    def display_process_info(
        self,
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
        iterations,
        travel_distance,
        find_resonances,
        min_a_axis,
    ) -> None:
        """Display process information."""
        real_travel_distance = travel_distance
        if self.kinematics == "corexy":
            # for the stepper to move `travel_distance` amount
            # in the logical axis, the real_travel_distance should be
            # divided by sqrt(2) as the head is going to move in both axes
            # in a corexy printer
            real_travel_distance = travel_distance / (2**0.5)
        if find_resonances:
            self.respond_info(
                f"Final max travel distance = {real_travel_distance:.2f} mm, "
                f"position min = {min_a_axis:.2f}, "
                f"traveling: {min_a_axis:.2f} --> {real_travel_distance + min_a_axis:.2f}"
            )
            self.respond_info(
                "Start find resonances mode, "
                f"speed: {min_speed:.2f}  --> {max_speed:.2f} mm/s with "
                f"{speed_change_step:.2f} step "
                f"current={current_min} mA "
                f"TBL={tbl_min} "
                f"TOFF={toff_min} "
                f"HSTRT={hstrt_min} "
                f"HEND={hend_min}"
            )
        else:
            self.respond_info(
                f"Final travel distance = {real_travel_distance:.2f} mm, "
                f"position min = {min_a_axis:.2f}, "
                f"traveling: {min_a_axis:.2f} --> {real_travel_distance + min_a_axis:.2f}"
            )
            self.respond_info(
                "Start of register enumeration mode, "
                f"speed: {min_speed:.2f}  --> {max_speed:.2f}  mm/s "
                f"current: {current_min} --> {current_max} mA "
                f"iterations={iterations} "
                f"TBL: {tbl_min} --> {tbl_max} "
                f"TOFF: {toff_min} --> {toff_max} "
                f"HSTRT: {hstrt_min} --> {hstrt_max} "
                f"HEND: {hend_min} --> {hend_max} "
                f"TPFD: {tpfd_min} --> {tpfd_max}"
            )

    def home_if_needed(self) -> None:
        """Home if not homed."""
        event_time = self.printer.get_reactor().monotonic()
        if "xyz" not in self.toolhead.get_status(event_time)["homed_axes"]:
            self.gcode.run_script_from_command("G28 X Y Z")
            self.toolhead.wait_moves()

    def measure_accelerometer_noise(self, accel_chip) -> str:
        """Measure accelerometer noise.

        Args:
            accel_chip (str): Accelerometer chip name, i.e adxl345.

        Returns:
            str: The measurement data file path.
        """
        start_time = time.time()
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
        # move the measurement file to the DATA_FOLDER
        measurement_data_path = accelerometer_measurement.move()
        self.respond_info(f"Noise Data: {measurement_data_path}")
        if self.debug:
            duration = time.time() - start_time
            self.respond_info(f"AccelerometerMeasure took {duration:0.1f} seconds")
        return measurement_data_path

    def measure_vibrations(
        self,
        axes,
        speed,
        travel_distance,
        travel_speed,
        max_speed,
        mid_a_axis,
        mid_b_axis,
        accel_chip,
        name,
        find_resonances,
    ) -> str:
        """Perform vibration measurement.

        Args:
            axes (list[str]): The main and secondary axis.
            speed (float): The speed for the measurement.
            travel_distance (float): The travel distance for the measurement.
            travel_speed (float): The travel speed for idle movements.
            max_speed (float): The maximum speed of the main axis.
            mid_a_axis (float): The middle position of the main axis.
            mid_b_axis (float): The middle position of the secondary axis.
            accel_chip (str): Accelerometer chip name, i.e adxl345.
            name (str): The name of the measurement.
            find_resonances (bool): Sets the mode to resonance measurement
                mode.

        Returns:
            str: The measurement data file path.
        """
        real_travel_distance = travel_distance
        if self.kinematics == "corexy":
            # for the stepper to move `travel_distance` amount
            # in the logical axis, the real_travel_distance should be
            # divided by sqrt(2) as the head is going to move in both axes
            # in a corexy printer
            real_travel_distance = real_travel_distance / (2**0.5)

        if find_resonances:
            # when finding resonances, keep the travel duration constant
            real_travel_distance = travel_distance * (speed / max_speed)
            self.gcode.run_script_from_command(f"G4 P{self.delay}")
            self.respond_info(
                f"Speed {speed:0.2f} mm/s on {real_travel_distance:0.2f} mm"
            )
            self.toolhead.wait_moves()

        # Start accel_chip data collection
        with AccelerometerMeasure(
            printer=self.printer,
            gcode=self.gcode,
            accel_chip=accel_chip,
            name=name,
        ) as accelerometer_measurement:
            if self.kinematics == "corexy":
                # isolate motors
                # move in logical axis
                if axes[0] == "x":
                    self.gcode.run_script_from_command(
                        f"G0 {axes[0]}{mid_a_axis + real_travel_distance} {axes[1]}{mid_b_axis + real_travel_distance} F{speed * 60}"
                    )
                elif axes[0] == "y":
                    self.gcode.run_script_from_command(
                        f"G0 {axes[0]}{mid_a_axis + real_travel_distance} {axes[1]}{mid_b_axis - real_travel_distance} F{speed * 60}"
                    )
            else:
                self.gcode.run_script_from_command(
                    f"G0 {axes[0]}{mid_a_axis + real_travel_distance} F{speed * 60}"
                )

        # Move to the initial position
        self.gcode.run_script_from_command(
            f"G0 {axes[0]}{mid_a_axis} {axes[1]}{mid_b_axis} F{travel_speed}"
        )
        self.toolhead.wait_moves()

        # move the measurement file to the DATA_FOLDER
        measurement_data_path = accelerometer_measurement.move()
        self.respond_info(f"Accel. data: {measurement_data_path}")

        return measurement_data_path


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
        iterations: int = 1,
        travel_distance: int | str = "default",
        accel_chip: str = "default",
        find_resonances: bool = False,
        run_plotter: bool = True,
    ) -> bool:
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
            iterations (int): Number of iterations, defaults to 1.
            travel_distance (int | str): The travel distance, or can be set to
                "default" to calculate the travel distance with the
                `measure_time`, `max_speed` and `accel_decel_distance`.
            accel_chip (str): The name of the acceleration chip.
            find_resonances (bool): Sets the mode to resonance measurement
                mode.
            run_plotter (bool): If set to True, the magnitude graphs will be
                generated after the vibration measurements are completed.

        Returns:
            bool: True if the command runs without any errors, False otherwise.
        """
        measure_time = self.measure_time / 1000
        self.reset_registers()
        # Find the steppers count of the main axis
        self.registers["stepper_count"] = self.get_stepper_count(axis)

        driver, sense_resistor = self.detect_driver(stepper=axis)
        self.validate_tpfd_values(driver, tpfd_min, tpfd_max)

        axes, steppers = self.get_axes_and_steppers(axis)

        (
            min_a_axis,
            max_a_axis,
            mid_a_axis,
            mid_b_axis,
            acceleration,
            travel_speed,
        ) = self.determine_axis_configuration(axes)

        accel_chip = self.get_accelerometer_chip(accel_chip)

        current_min, current_max = self.get_current_range(
            find_resonances, current_min, current_max, steppers
        )

        if find_resonances:
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
            find_resonances,
            measure_time,
            axes,
            steppers,
            min_a_axis,
            max_a_axis,
            acceleration,
        )

        travel_distance = self.calculate_travel_distance(
            axes,
            min_a_axis,
            max_a_axis,
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
            iterations,
            travel_distance,
            find_resonances,
            min_a_axis,
        )

        # Check for axis homing
        self.home_if_needed()

        self.gcode.run_script_from_command(f"SET_VELOCITY_LIMIT ACCEL={acceleration}")
        self.gcode.run_script_from_command(
            f"SET_VELOCITY_LIMIT ACCEL_TO_DECEL={acceleration}"
        )
        # Move to the initial position
        self.gcode.run_script_from_command(
            f"G0 {axes[0]}{mid_a_axis} {axes[1]}{mid_b_axis} F{travel_speed}"
        )
        self.toolhead.wait_moves()

        # Clean csv files
        self.clean_csv_files()

        # Measure accelerometer noise
        self.measure_accelerometer_noise(accel_chip)

        # Set steps of run_current
        start_coord = None
        if axes[0] == "x":
            start_coord = (mid_a_axis, mid_b_axis, 0)
        elif axes[0] == "y":
            start_coord = (mid_b_axis, mid_a_axis, 0)
        elif axes[0] == "z":
            start_coord = (mid_b_axis, 0, mid_a_axis)

        coord_generator = CoordGenerator(
            axes=axes,
            kinematics=self.kinematics,
            start_coord=Coord(start_coord)
        )

        for current in range(current_min, current_max + 1, self.current_change_step):
            self.apply_registers(steppers=steppers, field="curr", value=current)
            # Set tbl values
            for tbl in range(tbl_min, tbl_max + 1):
                self.apply_registers(steppers=steppers, field="tbl", value=tbl)
                # Set toff values
                for toff in range(toff_min, toff_max + 1):
                    self.apply_registers(steppers=steppers, field="toff", value=toff)
                    for hstrt_value in range(hstrt_min, hstrt_max + 1):
                        for hend_value in range(hend_min, hend_max + 1):
                            if (hend_value + hstrt_value) > hstrt_hend_max:
                                continue
                            # Set hend, and hstrt values
                            self.apply_registers(
                                steppers=steppers, field="hend", value=hend_value
                            )
                            self.apply_registers(
                                steppers=steppers, field="hstrt", value=hstrt_value
                            )
                            # Set tpfd values
                            for tpfd in range(tpfd_min, tpfd_max + 1):
                                if tpfd_min != -1 and tpfd_max != -1:
                                    self.apply_registers(
                                        steppers=steppers, field="tpfd", value=tpfd
                                    )
                                # Dump TMC settings
                                self.gcode.run_script_from_command(
                                    f"DUMP_TMC STEPPER={steppers[0]} REGISTER=chopconf"
                                )
                                freq = 1 / (
                                    2 * (12 + 32 * toff) * 1 / (1000000 * self.fclk)
                                    + 2 * 1 / (1000000 * self.fclk) * 16 * (1.5**tbl)
                                )
                                for speed in range(
                                    int(min_speed * 100),
                                    int(max_speed * 100) + 1,
                                    int(speed_change_step * 100),
                                ):
                                    speed = speed / 100
                                    for i in range(iterations):
                                        name = (
                                            f"__{current}_{tbl}_{toff}_{hstrt_value}_"
                                            f"{hend_value}_{tpfd}_{speed * 100:.0f}_"
                                            f"{freq:.0f}_{i + 1}__"
                                        )
                                        self.measure_vibrations(
                                            axes,
                                            speed,
                                            travel_distance,
                                            travel_speed,
                                            max_speed,
                                            mid_a_axis,
                                            mid_b_axis,
                                            accel_chip,
                                            name,
                                            find_resonances,
                                        )

        self.gcode.run_script_from_command("G4 P500")
        self.gcode.run_script_from_command(f"G0 {axis}{mid_a_axis} F{travel_speed}")
        self.toolhead.wait_moves()
        if run_plotter:
            self.respond_info("Magnitude graphs generation...")
            self.respond_info("This may take a while, please wait")
            # export data to processing
            self.gcode.run_script_from_command(
                f"RUN_SHELL_COMMAND CMD=chop_tune PARAMS='iterations={iterations} driver={driver} sense_resistor={sense_resistor}'"
            )
        # output data info
        self.respond_info(
            f"To run parser manually; type - RUN_SHELL_COMMAND CMD=chop_tune PARAMS='iterations={iterations} driver={driver} sense_resistor={sense_resistor}"
        )

        return True

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
            if IS_DIGIT.match(min_speed):
                min_speed = float(min_speed)
            max_speed = gcmd.get("MAX_SPEED", "default").lower()
            if IS_DIGIT.match(max_speed):
                min_speed = float(max_speed)
            speed_change_step = gcmd.get("SPEED_CHANGE_STEP", "default").lower()
            if IS_DIGIT.match(speed_change_step):
                speed_change_step = float(speed_change_step)
            iterations = int(gcmd.get("ITERATIONS", 1))
            travel_distance = gcmd.get("TRAVEL_DISTANCE", "default").lower()
            if IS_DIGIT.match(travel_distance):
                travel_distance = float(travel_distance)
            accel_chip = gcmd.get("ACCELEROMETER", "default").lower()
            find_resonances = {
                "0": False,
                "1": True,
                "false": False,
                "true": True,
            }.get(gcmd.get("FIND_RESONANCES", "false").lower(), False)
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
                iterations=iterations,
                travel_distance=travel_distance,
                accel_chip=accel_chip,
                find_resonances=find_resonances,
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
