from jax_mavrik.mavrik_types import State, Control 

# mavrik.py
from jax_mavrik.src.simulator import Simulator
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import ControlInputs, State
from jax_mavrik.mavrik_types import StateVariables, ControlInputs

import numpy as np 
from typing import Dict, Any, Tuple, Optional
from diffrax import ODETerm, Tsit5, diffeqsolve
import time
import os

import jax.numpy as jnp 

current_file_path = os.path.dirname(os.path.abspath(__file__))
mavrik_setup = MavrikSetup(file_path=os.path.join(current_file_path, "aero_export.mat"))

class Mavrik:
    def __init__(self, mass: float = 10.0, inertia: Tuple[float, float, float] = (0.5, 0.5, 0.8), dt: float = 0.01):
        self.simulator = Simulator(mass=mass, inertia=inertia, mavrik_setup=mavrik_setup)
        self.state_ndim = 27 
        self.control_ndim = 20
        self.state = None
        self.control = None
        self.dt = dt

    def reset(self, state: Optional[State]):
        assert state.shape == (self.state_ndim,)
        self.state = StateVariables(*state)
      
    def step(self, control: Control) -> jnp.ndarray:
        if self.state is None or control is None:
            raise ValueError("State and control must be initialized using reset() before calling step().")
        assert control.shape == (self.control_ndim,)
        control_input = ControlInputs(*control)
        self.state = self.simulator.run(self.state, control_input, self.dt)

        return np.array(self.state._asdict().values())
    

# Example usage
if __name__ == "__main__":
    
    initial_state = np.array([
        10.0, 0.0, 0.0,  # Vx, Vy, Vz
        0.0, 0.0, 0.0,   # X, Y, Z
        0.0, 0.0, 0.0,   # roll, pitch, yaw
        0.0, 0.0, 0.0,   # Vbx, Vby, Vbz
        0.0, 0.0, 0.0,   # wx, wy, wz
        0.0, 0.0, 0.0,   # dwdt_x, dwdt_y, dwdt_z
        0.0, 0.0, 0.0,   # ax, ay, az
        0.0, 0.0, 0.0,   # Fx, Fy, Fz
        0.0, 0.0, 0.0    # L, M, N
    ])

    control = np.array([
        0.0, 0.0, 0.0,  # wing_tilt, tail_tilt, aileron
        0.0, 0.0, 0.0,  # elevator, flap, rudder
        1000.0, 1000.0,  # RPM_tailLeft, RPM_tailRight
        1000.0, 1000.0,  # RPM_leftOut1, RPM_left2
        1000.0, 1000.0,  # RPM_left3, RPM_left4
        1000.0, 1000.0,  # RPM_left5, RPM_left6In
        1000.0, 1000.0,  # RPM_right7In, RPM_right8
        1000.0, 1000.0,  # RPM_right9, RPM_right10
        1000.0, 1000.0   # RPM_right11, RPM_right12Out
    ])

    mavrik = Mavrik()
    mavrik.reset(initial_state)

    num_steps = 10
    states = [initial_state]
    start_time = time.time()
    for _ in range(num_steps):
        state = mavrik.step(control)
        states.append(state)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"[Iteration runtime] Tot: {runtime:.6f} seconds | Avg: {runtime / num_steps:.6f} seconds")

    print("States:", states)