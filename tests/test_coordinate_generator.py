"""Test CoordinateGenerator class."""

# Third-Party Imports
import pytest

# Local Imports
from chopper_tune import Coord, CoordGenerator


@pytest.mark.parametrize(
    "axes,kinematics,start_coord,expected_result", (
        # corexy
        (("x", "y"), "corexy", Coord((0, 0, 0)), Coord(((2**0.5)/2, (2**0.5)/2, 0))),

        # cartesian
        (("x", "y"), "cartesian", Coord((0, 0, 0)), Coord((1, 0, 0))),
    )
)
def test_initial_direction(axes, kinematics, start_coord, expected_result):
    """test initial_direction is properly set."""
    coord = CoordGenerator(
        axes=axes,
        kinematics=kinematics,
        start_coord=start_coord,
    )
    assert coord.direction.x == pytest.approx(expected_result.x)
    assert coord.direction.y == pytest.approx(expected_result.y)
    assert coord.direction.z == pytest.approx(expected_result.z)
    assert coord.direction.length() == pytest.approx(1.0)


def test_next_position_core_xy():
    """Generate coordinates for CoreXY."""
    generator = CoordGenerator(
        axes = ["x", "y"],
        kinematics="corexy",
        start_coord=Coord((0, 0, 0)),
    )
    assert generator.direction is not None

    pos1 = generator.next(1.0)
    assert isinstance(pos1, Coord)
    assert pos1.x == pytest.approx((2**0.5)/2)
    assert pos1.y == pytest.approx((2**0.5)/2)
    assert pos1.z == pytest.approx(0)

    pos2 = generator.next(10.0)
    assert pos2.x == pytest.approx((2**0.5) / 2 - (10*(2**0.5) / 2))
    assert pos2.y == pytest.approx((2**0.5) / 2 - (10*(2**0.5) / 2))
    assert pos2.z == pytest.approx(0)

    pos3 = generator.next(9.0)
    assert pos3.x == pytest.approx(0)
    assert pos3.y == pytest.approx(0)
    assert pos3.z == pytest.approx(0)
