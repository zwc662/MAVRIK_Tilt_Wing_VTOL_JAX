from typing import Literal, Union, List, Tuple
from jaxtyping import Array, Float
import jax.numpy as jnp
from dataclasses import dataclass


arr = Union[jnp.ndarray, Array]

PLANE_STATE_NX = 13
FULL_STATE_NX = 16
OUTER_CONTROL_NU = 4
INNER_CONTROL_NU = 4

PlaneState = Float[arr, "12"]
FullState = Float[arr, "12"]


# [tilt]
OuterControl = Float[arr, "1"]

# [throt, ele, ail, rud]
InnerControl = Float[arr, "4"]

# [tilt, throt, ail, ele, rud]
InnerOuterControl = Float[arr, "5"]
 

MavrikModelType = Literal["morelli"]


class S:
    VT = 0
    ALPHA = 1
    BETA = 2
    #
    PHI = 3
    THETA = 4
    PSI = 5
    #
    P = 6
    Q = 7
    R = 8
    #
    ALT = 11
    POWER = 12

    PQR = slice(6, 9)


class C:
    THROTTLE = 0
    EL = 1
    AIL = 2
    RDR = 3


@dataclass
class ControlInputs:
    tilt_angles: Tuple[float, float]  # Wing and tail tilt angles
    rpms: List[float]  # Engine/propeller RPMs
    control_surfaces: Tuple[float, float, float]  # Aileron, elevator, rudder

@dataclass
class State:
    position: jnp.ndarray  # [x, y, z] in NED frame
    velocity: jnp.ndarray  # [u, v, w] in body frame
    orientation: jnp.ndarray  # [phi, theta, psi] Euler angles
    angular_rates: jnp.ndarray  # [p, q, r] in body frame