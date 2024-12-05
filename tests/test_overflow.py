import os
import sys
import numpy as np
import pytest
from jax_mavrik.mavrik import Mavrik
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs
 
  
vned_values = np.array([3152.88807, 2404.48661, -445.080451])
xned_values = np.array([14.6961720, -9.39669706, -20.3385467])
vb_values = np.array([-159.01554784, -1443.88400606, -1887.40120276])
euler_values = np.array([1.77307682, -0.70741363, -0.67015054])
pqr_values = np.array([-515.238191, 53.5245398, 5.55108337])
state = np.array([
    3152.88807, 2404.48661, -445.080451, 14.6961720, -9.39669706, -20.3385467, 
    -159.015548, -1443.88401, -1887.40120, 1.77307682, -0.707413627, -0.670150537, 
    -515.238191, 53.5245398, 5.55108337, 73791.6996, 70404.9672, 126237.362, 
    -101300.236, 49342.9484, 20063.5134
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
 

@pytest.fixture
def mavrik():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    return Mavrik(mavrik_setup=mavrik_setup, dt=0.01, method='RK4', fixed_step_size=0.01)
 
 
def test_simulation_output(mavrik): 
    for t in range(10):
        nxt_state, info = mavrik.step(state, control)
        nxt_vned = nxt_state[mavrik.STATE.VXe:mavrik.STATE.VXe+3]
        nxt_xned = nxt_state[mavrik.STATE.Xe:mavrik.STATE.Xe+3]
        nxt_vb = nxt_state[mavrik.STATE.u:mavrik.STATE.u+3]
        nxt_euler = nxt_state[mavrik.STATE.roll:mavrik.STATE.roll+3]
        nxt_pqr = nxt_state[mavrik.STATE.p:mavrik.STATE.p+3]
        forces = nxt_state[mavrik.STATE.Fx:mavrik.STATE.Fx+3]
        moments = nxt_state[mavrik.STATE.L:mavrik.STATE.L+3]
        

        print(f"Next VNED @ {t}: {nxt_vned}")
        print(f"Next XNED @ {t}: {nxt_xned}")
        print(f"Next VB @ {t}: {nxt_vb}")
        print(f"Next Euler @ {t}: {nxt_euler}")
        print(f"Next PQR @ {t}: {nxt_pqr}")
        print(f"Forces @ {t}: {forces}")
        print(f"Moments @ {t}: {moments}")
