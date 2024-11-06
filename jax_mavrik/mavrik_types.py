from typing import Literal, Union, List, Tuple
from jaxtyping import Array, Float
import jax.numpy as jnp 
import numpy as np

from typing import NamedTuple


arr = Union[jnp.ndarray, Array]
State = Float[arr, "28"]   
Control = Float[arr, "20"]


  
class ControlInputs(NamedTuple):
    RPM_tailLeft: Float
    RPM_tailRight: Float
    RPM_leftOut1: Float
    RPM_left2: Float
    RPM_left3: Float
    RPM_left4: Float
    RPM_left5: Float
    RPM_left6In: Float
    RPM_right7In: Float
    RPM_right8: Float
    RPM_right9: Float
    RPM_right10: Float
    RPM_right11: Float
    RPM_right12Out: Float
    wing_tilt: Float
    tail_tilt: Float
    aileron: Float
    elevator: Float
    flap: Float
    rudder: Float

  

class StateVariables(NamedTuple):  
    Vx: Float  # Velocity in x direction
    Vy: Float  # Velocity in y direction
    Vz: Float  # Velocity in z direction
    X: Float  # Position in x direction
    Y: Float  # Position in y direction
    Z: Float  # Position in z direction
    roll: Float  # Roll angle
    pitch: Float  # Pitch angle
    yaw: Float  # Yaw angle
    # DCM: np.ndarray  # Direction Cosine Matrix, np.array([[3x3]]) # If using xyz coordinate, there is no need for NED frame
    Vbx: Float  # Body-frame velocity in x direction
    Vby: Float  # Body-frame velocity in y direction
    Vbz: Float  # Body-frame velocity in z direction
    wx: Float  # Angular velocity in x direction
    wy: Float  # Angular velocity in y direction
    wz: Float  # Angular velocity in z direction
    dwdt_x: Float  # Angular acceleration in x direction
    dwdt_y: Float  # Angular acceleration in y direction
    dwdt_z: Float  # Angular acceleration in z direction
    ax: Float  # Linear acceleration in x direction
    ay: Float  # Linear acceleration in y direction
    az: Float  # Linear acceleration in z direction
    Fx: Float  # Force in x direction
    Fy: Float  # Force in y direction
    Fz: Float  # Force in z direction
    L: Float  # Moment about x-axis
    M: Float  # Moment about y-axis
    N: Float  # Moment about z-axis


class AeroState(NamedTuple):
    Vx: Float
    Vy: Float
    Vz: Float
    roll: Float
    pitch: Float
    yaw: Float
    p: Float
    q: Float
    r: Float

class Forces(NamedTuple):
    Fx: Float
    Fy: Float
    Fz: Float

class Moments(NamedTuple):
    L: Float
    M: Float
    N: Float