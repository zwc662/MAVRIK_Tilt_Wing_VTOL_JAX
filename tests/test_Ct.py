
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import expected_actuator_outputs_values as actuator_outputs_values, expected_Ct_outputs_values as expected_Ct_outputs_values
 

@pytest.fixture
def mavrik_aero():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    return MavrikAero(mavrik_setup=mavrik_setup)


@pytest.mark.parametrize(
    "id, actuator_outputs_values, expected_Ct_outputs_values",
    zip(list(range(11)), actuator_outputs_values, expected_Ct_outputs_values)
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, expected_Ct_outputs_values):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    
    F0, M0 = mavrik_aero.Ct(u)
    F0_array = jnp.array([F0.Fx, F0.Fy, F0.Fz])
    M0_array = jnp.array([M0.L, M0.M, M0.N])
    Ct_outputs_values = jnp.concatenate([F0_array, M0_array])
    Ct_outputs_values_close = jnp.allclose(Ct_outputs_values, expected_Ct_outputs_values, atol=0.001)
    print("Ct_outputs_values_close???", Ct_outputs_values_close)
    if not Ct_outputs_values_close:
        print(f"\n  Expected: {expected_Ct_outputs_values}\n  Got: {Ct_outputs_values}")
        max_diff_index_Ct_outputs_values = jnp.argmax(jnp.abs(Ct_outputs_values - expected_Ct_outputs_values))
        print(f"\n  Max difference in Ct_outputs_values at index {max_diff_index_Ct_outputs_values}: Expected {expected_Ct_outputs_values[max_diff_index_Ct_outputs_values]}, Got {Ct_outputs_values[max_diff_index_Ct_outputs_values]}")
    
     