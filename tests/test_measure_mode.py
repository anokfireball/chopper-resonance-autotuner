"""MeasureMode related tests are here."""
# Standard Library Imports
import sys
from enum import IntEnum

# Third-Party Imports
import pytest

# Local Imports
from chopper_tune import MeasureMode


@pytest.mark.parametrize(
    "mode",
    [
        MeasureMode.Resonances,
        MeasureMode.Vibrations,
    ],
)
def test_it_is_an_int_enum(mode):
    """MeasureMode is an IntEnum."""
    assert isinstance(mode, IntEnum)


@pytest.mark.parametrize(
    "mode,expected_value",
    [
        [MeasureMode.Resonances, 0],
        [MeasureMode.Vibrations, 1],
    ],
)
def test_enum_values(mode, expected_value):
    """Test enum values."""
    assert mode == expected_value


@pytest.mark.parametrize(
    "mode,expected_value",
    [
        [MeasureMode.Resonances, "Resonances"],
        [MeasureMode.Vibrations, "Vibrations"],
    ],
)
def test_enum_names(mode, expected_value):
    """Test enum names."""
    assert str(mode) == expected_value


def test_to_mode_mode_is_skipped():
    """MeasureMode.to_mode() mode is skipped."""
    with pytest.raises(TypeError) as cm:
        _ = MeasureMode.to_mode()

    py_error_message = {
        8: "to_mode() missing 1 required positional argument: 'mode'",
        9: "to_mode() missing 1 required positional argument: 'mode'",
    }.get(
        sys.version_info.minor,
        "MeasureMode.to_mode() missing 1 required positional argument: 'mode'"
    )
    assert str(cm.value) == py_error_message


def test_to_mode_mode_is_none():
    """MeasureMode.to_mode() mode is None."""
    with pytest.raises(TypeError) as cm:
        _ = MeasureMode.to_mode(None)
    assert str(cm.value) == (
        "mode should be a MeasureMode enum value or one "
        "of ['Resonances', 'Vibrations', 0, 1], not NoneType: 'None'"
    )


def test_to_mode_mode_is_not_a_str():
    """MeasureMode.to_mode() mode is not an int or str."""
    with pytest.raises(TypeError) as cm:
        _ = MeasureMode.to_mode(12334.123)

    assert str(cm.value) == (
        "mode should be a MeasureMode enum value or one of "
        "['Resonances', 'Vibrations', 0, 1], not float: '12334.123'"
    )


def test_to_mode_mode_is_not_a_valid_str():
    """MeasureMode.to_mode() mode is not a valid str."""
    with pytest.raises(ValueError) as cm:
        _ = MeasureMode.to_mode("not a valid value")

    assert str(cm.value) == (
        "mode should be a MeasureMode enum value or one of "
        "['Resonances', 'Vibrations', 0, 1], not 'not a valid value'"
    )


@pytest.mark.parametrize(
    "mode_name,mode",
    [
        # Resonances
        ["Resonances", MeasureMode.Resonances],
        ["resonances", MeasureMode.Resonances],
        ["RESONANCES", MeasureMode.Resonances],
        ["ReSoNaNcEs", MeasureMode.Resonances],
        ["rEsOnAnCeS", MeasureMode.Resonances],
        [0, MeasureMode.Resonances],
        # Vibrations
        ["Vibrations", MeasureMode.Vibrations],
        ["vibrations", MeasureMode.Vibrations],
        ["VIBRATIONS", MeasureMode.Vibrations],
        ["ViBrAtIoNs", MeasureMode.Vibrations],
        ["vIbRaTiOnS", MeasureMode.Vibrations],
        [1, MeasureMode.Vibrations],
    ],
)
def test_to_mode_is_working_properly(mode_name, mode):
    """MeasureMode can parse schedule mode names."""
    assert MeasureMode.to_mode(mode_name) == mode
