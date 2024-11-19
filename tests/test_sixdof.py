import os
import sys
import pytest
from jax_mavrik.src.sixdof import SixDOFDynamics, RigidBody, SixDOFState
from jax_mavrik.mavrik_setup import MavrikSetup 

from .test_mavrik import vned_values, xned_values, euler_values, vb_values, ab_values, pqr_values, dotpqr_values, forces_values, moments_values
import jax.numpy as jnp
 
@pytest.fixture
def rigid_body():
    mavrik_setup = MavrikSetup(file_path=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "jax_mavrik/aero_export.mat")
    )
    mass = mavrik_setup.mass
    inertia = mavrik_setup.inertia
    return RigidBody(mass=mass, inertia=inertia)


@pytest.mark.parametrize(
    "id, vned, xned, euler, vb, pqr, forces, moments, \
        expected_vned, expected_xned, expected_euler, expected_vb, expected_pqr, expected_ab, expected_dotpqr",
    zip(
        list(range(10)),
        vned_values[:-1], xned_values[:-1], euler_values[:-1], vb_values[:-1], pqr_values[:-1], forces_values[:-1], moments_values[:-1],
        vned_values[1:], xned_values[1:], euler_values[1:], vb_values[1:], pqr_values[1:], ab_values[1:], dotpqr_values[1:]
    )    
)

def test_sixdof(id, rigid_body, vned, xned, euler, vb, pqr, forces, moments, 
                expected_vned, expected_xned, expected_euler, expected_vb, expected_pqr, expected_ab, expected_dotpqr):
    initial_state = SixDOFState(
        Ve=vned,
        Xe=xned, 
        Vb=vb,
        Euler=euler,
        pqr=pqr,
        ab=[0., 0., 0.],
        dotpqr=[0., 0., 0.]
    )

    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    
    dynamics = SixDOFDynamics(rigid_body, method="euler", fixed_step_size=0.01)
    results = dynamics.run_simulation(initial_state, forces, moments, 0, 0.01)
    nxt_state = results["states"][-1]
    nxt_vned = nxt_state[:3]
    nxt_xned = nxt_state[3:6]
    nxt_vb = nxt_state[6:9]
    nxt_euler = nxt_state[9:12]
    nxt_pqr = nxt_state[12:15]
    nxt_ab = nxt_state[15:18]
    nxt_dotpqr = nxt_state[18:21]
    threshold = 0.1
    
    print(f"Ve Expected: {expected_vned}, Got: {nxt_vned}, Close: {jnp.allclose(expected_vned, nxt_vned, atol=threshold)}")
    print(f"Xe Expected: {expected_xned}, Got: {nxt_xned}, Close: {jnp.allclose(expected_xned, nxt_xned, atol=threshold)}")
    print(f"Vb Expected: {expected_vb}, Got: {nxt_vb}, Close: {jnp.allclose(expected_vb, nxt_vb, atol=threshold)}")
    print(f"Euler Expected: {expected_euler}, Got: {nxt_euler}, Close: {jnp.allclose(expected_euler, nxt_euler, atol=threshold)}")
    print(f"pqr Expected: {expected_pqr}, Got: {nxt_pqr}, Close: {jnp.allclose(expected_pqr, nxt_pqr, atol=threshold)}")
    print(f"ab Expected: {expected_ab}, Got: {nxt_ab}, Close: {jnp.allclose(expected_ab, nxt_ab, atol=threshold)}")
    print(f"dotpqr Expected: {expected_dotpqr}, Got: {nxt_dotpqr}, Close: {jnp.allclose(expected_dotpqr, nxt_dotpqr, atol=threshold)}")
     