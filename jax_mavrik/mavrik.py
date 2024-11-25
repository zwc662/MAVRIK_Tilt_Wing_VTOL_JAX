# mavrik.py
from jax_mavrik.src.simulator import Simulator
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import ControlInputs, StateArr, ControlArr
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, MAVRIK_STATE, MAVRIK_CONTROL
from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import numpy as np 
from typing import Dict, Any, Tuple, Optional, Union
from diffrax import ODETerm, Tsit5, diffeqsolve
import time
import os

import jax.numpy as jnp 
import jax 

current_file_path = os.path.dirname(os.path.abspath(__file__))
_MAVRIK_SETUP_ = MavrikSetup(file_path=os.path.join(current_file_path, "aero_export.mat"))

class Mavrik:
    def __init__(self, mavrik_setup: Optional[MavrikSetup] = None, dt: float = 0.01, method: str = 'RK4', fixed_step_size: float = 0.01):
        mavrik_setup = mavrik_setup if mavrik_setup is not None else _MAVRIK_SETUP_
        self.simulator = Simulator(mavrik_setup=mavrik_setup, method = method, fixed_step_size=fixed_step_size)
        self.state_ndim = 21
        self.control_ndim = 20
        self.control = None 
        self.dt = dt
        self.ned2xyz = lambda euler, ned: euler_to_dcm(*euler) @ ned
        self.STATE = MAVRIK_STATE
        self.CONTROL = MAVRIK_CONTROL
 
    def step(self, state: StateArr, control: ControlArr): # -> jnp.ndarray:
        if state is None or control is None:
            raise ValueError("State and control must be initialized using reset() before calling step().")
        assert control.shape == (self.control_ndim,), f"Control shape mismatch. Expected: {self.control_ndim}, Got: {control.shape}"
        assert state.shape == (self.state_ndim,), f"State shape mismatch. Expected: {self.state_ndim}, Got: {state.shape}"
        cur_state = StateVariables(*state)
        control_input = ControlInputs(*control)
        nxt_state, info= self.simulator.run(cur_state, control_input, self.dt)
        info['control'] = control_input
        return np.asarray(list(nxt_state._asdict().values())), info
     
# Example usage
if __name__ == "__main__":
     

    ## vned=state[:3]
    expected_vned = np.array([
        [29.9568, -0.0000, -0.0997],
        [29.9143, -0.0000, -0.1922],
        [29.8730, -0.0001, -0.2767],
        [29.8329, -0.0002, -0.3529],
        [29.7944, -0.0004, -0.4203],
        [29.7574, -0.0006, -0.4789],
        [29.7222, -0.0009, -0.5286],
        [29.6886, -0.0014, -0.5692],
        [29.6565, -0.0019, -0.6011],
        [29.6261, -0.0024, -0.6243]
    ])

    # xned = state[:3]
    expected_xned = np.array([
        [0.2998, -0.0000, -0.0005],
        [0.5991, -0.0000, -0.0020],
        [0.8981, -0.0000, -0.0043],
        [1.1966, -0.0000, -0.0075],
        [1.4947, -0.0000, -0.0114],
        [1.7925, -0.0000, -0.0159],
        [2.0899, -0.0000, -0.0209],
        [2.3869, -0.0000, -0.0264],
        [2.6837, -0.0000, -0.0323],
        [2.9801, -0.0001, -0.0384]
    ])

    expected_vb = np.array([
        [29.8923, -0.0001, 1.9667],
        [29.8605, -0.0004, 1.8037],
        [29.8309, -0.0010, 1.6083],
        [29.8028, -0.0017, 1.3853],
        [29.7755, -0.0027, 1.1393],
        [29.7484, -0.0038, 0.8750],
        [29.7209, -0.0053, 0.5969],
        [29.6924, -0.0069, 0.3092],
        [29.6626, -0.0089, 0.0163],
        [29.6313, -0.0111, -0.2781]
    ])

    expected_euler = np.array([
        [-0.0000, 0.0690, 0.0000],
        [-0.0002, 0.0668, 0.0000],
        [-0.0003, 0.0631, 0.0000],
        [-0.0006, 0.0583, 0.0000],
        [-0.0010, 0.0524, 0.0000],
        [-0.0014, 0.0455, 0.0001],
        [-0.0019, 0.0379, 0.0001],
        [-0.0025, 0.0296, 0.0002],
        [-0.0032, 0.0208, 0.0002],
        [-0.0040, 0.0117, 0.0003]
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
   
    state = np.array([
        30.0000, 0, 0,  # Xe Ye Ze
        0.0000, 0.0000, 0.0000,  # VXe, VYe, VZe
        29.9269, 0, 2.0927, # u v w
        0, 0.0698, 0,   # roll, pitch, yaw 
        0.0, 0.0, 0.0,   # p, q, r
        0.0, 0.0, 0.0,   # Fx, Fy, Fz
        0.0, 0.0, 0.0    # L, M, N
    ])

    

    mavrik = Mavrik(method = 'rk4')
    
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
        print(f">>>>>>>>>>>>>>>>>>>> Iteration: {i} <<<<<<<<<<<<<<<<<<<<<<")
        print(f"Runtime: {runtime:.6f} | Tot: {tot_runtime:.6f} seconds | Avg: {tot_runtime / num_steps:.6f} seconds | State: {state}")
        vned = state[MAVRIK_STATE.VXe:MAVRIK_STATE.VXe+3]
        print(f"Vned: {vned} | Expected Vned: {expected_vned[i]} | Error: {np.linalg.norm(vned - expected_vned[i])}")
        xned = state[MAVRIK_STATE.Xe:MAVRIK_STATE.Xe+3]
        print(f"Xned: {xned} | Expected Xned: {expected_xned[i]} | Error: {np.linalg.norm(xned - expected_xned[i])}")
        vb = state[MAVRIK_STATE.u:MAVRIK_STATE.u+3]
        print(f"Vb: {vb} | Expected Vb: {expected_vb[i]} | Error: {np.linalg.norm(vb - expected_vb[i])}")
        euler = state[MAVRIK_STATE.roll:MAVRIK_STATE.roll+3]
        print(f"Euler: {euler} | Expected Euler: {expected_euler[i]} | Error: {np.linalg.norm(euler - expected_euler[i])}")
         
        expected_dcm = euler_to_dcm(*expected_euler[i])
        vned_transformed = expected_dcm @ expected_vned[i]
        print(f"DCM @ Expected Vned: {vned_transformed} | Expected Vb: {expected_vb[i]} | Error: {np.linalg.norm(vned_transformed - expected_vb[i])}")
    #print("States:", states)