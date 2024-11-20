from typing import Literal, Union, List, Tuple
from jaxtyping import Array, Float
import jax.numpy as jnp 
import numpy as np

from typing import NamedTuple
from enum import IntEnum


arr = Union[jnp.ndarray, Array]
StateArr = Float[arr, "21"]   
ControlArr = Float[arr, "20"]


  
class ControlInputs(NamedTuple):
    wing_tilt: Float
    tail_tilt: Float
    aileron: Float
    elevator: Float
    flap: Float
    rudder: Float
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
    
class MAVRIK_STATE(IntEnum):
    VXe = 0
    VYe = 1
    VZe = 2
    Xe = 3
    Ye = 4
    Ze = 5
    u = 6
    v = 7
    w = 8
    roll = 9
    pitch = 10
    yaw = 11
    p = 12
    q = 13
    r = 14
    Fx = 15
    Fy = 16
    Fz = 17
    L = 18
    M = 19
    N = 20

class StateVariables(NamedTuple):  
    VXe: Float  # Velocity in NED frame1
    VYe: Float  # Velocity in NED frame2
    VZe: Float  # Velocity in NED frame3

    Xe: Float  # Position in x direction
    Ye: Float  # Position in y direction
    Ze: Float  # Position in z direction

    u: Float  # Body-frame velocity in x direction
    v: Float  # Body-frame velocity in y direction
    w: Float  # Body-frame velocity in z direction

    roll: Float  # Roll angle
    pitch: Float  # Pitch angle
    yaw: Float  # Yaw angle
    # DCM: np.ndarray  # Direction Cosine Matrix, np.array([[3x3]]) # If using xyz coordinate, there is no need for NED frame
   
    p: Float  # Angular velocity in x direction
    q: Float  # Angular velocity in y direction
    r: Float  # Angular velocity in z direction
    #pdot: Float  # Angular acceleration in x direction
    #qdot: Float  # Angular acceleration in y direction
    #rdot: Float  # Angular acceleration in z direction
    #udot: Float  # Linear acceleration in x direction
    #vdot: Float  # Linear acceleration in y direction
    #wdot: Float  # Linear acceleration in z direction
    Fx: Float  # Force in x direction
    Fy: Float  # Force in y direction
    Fz: Float  # Force in z direction
    L: Float  # Moment about x-axis
    M: Float  # Moment about y-axis
    N: Float  # Moment about z-axis


class AeroState(NamedTuple):
    VXe: Float
    VYe: Float
    VZe: Float
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