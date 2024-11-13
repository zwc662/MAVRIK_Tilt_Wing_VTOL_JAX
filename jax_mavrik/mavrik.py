# mavrik.py
from jax_mavrik.src.simulator import Simulator
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import ControlInputs, StateArr, ControlArr
from jax_mavrik.mavrik_types import StateVariables, ControlInputs

import numpy as np 
from typing import Dict, Any, Tuple, Optional, Union
from diffrax import ODETerm, Tsit5, diffeqsolve
import time
import os

import jax.numpy as jnp 

current_file_path = os.path.dirname(os.path.abspath(__file__))
_MAVRIK_SETUP_ = MavrikSetup(file_path=os.path.join(current_file_path, "aero_export.mat"))

class Mavrik:
    def __init__(self, mavrik_setup: Optional[MavrikSetup] = None, dt: float = 0.01, method: str = 'RK4', fixed_step_size: float = 0.01):
        mavrik_setup = mavrik_setup if mavrik_setup is not None else _MAVRIK_SETUP_
        self.simulator = Simulator(mavrik_setup=mavrik_setup, method = method, fixed_step_size=fixed_step_size)
        self.state_ndim = 27 
        self.control_ndim = 20
        self.control = None 
        self.dt = dt
 
    def step(self, state: StateArr, control: ControlArr): # -> jnp.ndarray:
        if state is None or control is None:
            raise ValueError("State and control must be initialized using reset() before calling step().")
        assert control.shape == (self.control_ndim,)
        assert state.shape == (self.state_ndim,)
        cur_state = StateVariables(*state)
        control_input = ControlInputs(*control)
        nxt_state = self.simulator.run(cur_state, control_input, self.dt)

        info = {'state': cur_state, 'control': control_input, 'nxt_state': nxt_state}
        return np.asarray(list(nxt_state._asdict().values())), info
     
# Example usage
if __name__ == "__main__":
     
    U = 30  # trim speed
    eulerIn = [0,0.069813,0]  # trim attitude (roll, pitch, yaw)
    vnedIn = [30., 0., 0.]  # NED velocity
    # Convert NED velocity to body frame velocity
   
    state = np.array([
        *vnedIn,  # u, v, w
        0.0, 0.0, 0.0,   # X, Y, Z
        *eulerIn,   # roll, pitch, yaw
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
        7500.0, 7500.0,  # RPM_tailLeft, RPM_tailRight
        7500.0, 7500.0,  # RPM_leftOut1, RPM_left2
        7500.0, 7500.0,  # RPM_left3, RPM_left4
        7500.0, 7500.0,  # RPM_left5, RPM_left6In
        7500.0, 7500.0,  # RPM_right7In, RPM_right8
        7500.0, 7500.0,  # RPM_right9, RPM_right10
        7500.0, 7500.0   # RPM_right11, RPM_right12Out
    ])

    mavrik = Mavrik()
    
    num_steps = int(0.1 / 0.01)
    states = [state]
    tot_runtime = 0.0
    for i in range(num_steps):
        start_time = time.time()
        state, info = mavrik.step(state, control)
        end_time = time.time()
        runtime = end_time - start_time
        tot_runtime += runtime
        states.append(state)
        print(f"[Iteration {i}] Runtime: {runtime:.6f} | Tot: {tot_runtime:.6f} seconds | Avg: {tot_runtime / num_steps:.6f} seconds | State.: {info['state']}")
    

    #print("States:", states)