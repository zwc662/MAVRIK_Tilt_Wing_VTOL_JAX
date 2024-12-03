
import pytest

import os
import sys

from jax_mavrik.src.mavrik_aero import MavrikAero, Ct, interpolate_nd, CT_LOOKUP_TABLES, RPM_TRANSFORMS

from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import StateVariables, ControlInputs, Forces
from jax_mavrik.src.actuator import ActuatorOutput 

from jax_mavrik.src.utils.mat_tools import euler_to_dcm

import jax.numpy as jnp

from .test_mavrik_aero import mavrik_aero, expected_actuator_outputs_values as actuator_outputs_values, expected_Ct_outputs_values as expected_Ct_outputs_values
 
@pytest.mark.parametrize(
    "id, actuator_outputs_values, expected_Ct_outputs_values",
    zip(list(range(11)), actuator_outputs_values, expected_Ct_outputs_values)
)
 
def test_mavrik_aero(id, mavrik_aero, actuator_outputs_values, expected_Ct_outputs_values):
    u = ActuatorOutput(*actuator_outputs_values)
        
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    

    wing_transform = jnp.array([[jnp.cos(u.wing_tilt), 0, jnp.sin(u.wing_tilt)], [0, 1, 0], [-jnp.sin(u.wing_tilt), 0., jnp.cos(u.wing_tilt)]])
    tail_transform = jnp.array([[jnp.cos(u.tail_tilt), 0, jnp.sin(u.tail_tilt)], [0, 1, 0], [-jnp.sin(u.tail_tilt), 0., jnp.cos(u.tail_tilt)]])
 
    F0, M0 = mavrik_aero.Ct(u, wing_transform, tail_transform)
    F0_array = jnp.array([F0.Fx, F0.Fy, F0.Fz])
    M0_array = jnp.array([M0.L, M0.M, M0.N])
    Ct_outputs_values = jnp.concatenate([F0_array, M0_array])
    Ct_outputs_values_close = jnp.allclose(Ct_outputs_values, expected_Ct_outputs_values, atol=0.001)
    print("Ct_outputs_values_close???", Ct_outputs_values_close)
    if not Ct_outputs_values_close:
        print(f"\n  Expected: {expected_Ct_outputs_values}\n  Got: {Ct_outputs_values}")
        max_diff_index_Ct_outputs_values = jnp.argmax(jnp.abs(Ct_outputs_values - expected_Ct_outputs_values))
        print(f"\n  Max difference in Ct_outputs_values at index {max_diff_index_Ct_outputs_values}: Expected {expected_Ct_outputs_values[max_diff_index_Ct_outputs_values]}, Got {Ct_outputs_values[max_diff_index_Ct_outputs_values]}")
    
     