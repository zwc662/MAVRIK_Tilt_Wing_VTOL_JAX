import os
import sys
import pytest
from jax_mavrik.src.sixdof import SixDOFDynamics, RigidBody, SixDOFState
from jax_mavrik.mavrik_setup import MavrikSetup 
from jax_mavrik.src.utils.mat_tools import euler_to_dcm

from .test_mavrik import vned_values, xned_values, euler_values, dcm_values, vb_values, pqr_values, forces_values, moments_values
import jax.numpy as jnp
from jax import vmap
 
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
        expected_vned, expected_xned, expected_euler, expected_dcm, expected_vb, expected_pqr",
    zip(
        list(range(10)),
        vned_values[:-1], xned_values[:-1], euler_values[:-1], vb_values[:-1], pqr_values[:-1], forces_values[:-1], moments_values[:-1],
        vned_values[1:], xned_values[1:], euler_values[1:], dcm_values[:-1], vb_values[1:], pqr_values[1:]
    )    
)
 

def test_sixdof(id, rigid_body, vned, xned, euler, vb, pqr, forces, moments, 
                expected_vned, expected_xned, expected_euler, expected_dcm, expected_vb, expected_pqr):
    print(f">>>>>>>>>>>>>>>>>>>> Test ID: {id} <<<<<<<<<<<<<<<<<<<<<<")
    threshold = 0.001

    dcm = euler_to_dcm(*euler)
    dcm_close = jnp.allclose(dcm.flatten(), expected_dcm.flatten(), atol=threshold)
    print(f"DCM Expected: {expected_dcm}, Got: {dcm}, Close: {dcm_close}, Max Error: {jnp.max(jnp.abs(dcm.flatten() - expected_dcm.flatten()))}") 

    uvw =  dcm @ vned
    vb_close = jnp.allclose(vb, uvw, atol=threshold) 
    print(f"Ve=>Vb Expected: {vb}, Got: {uvw}, Close: {vb_close}, Max Error: {jnp.max(jnp.abs(vb - uvw))}") 
    #if not vb_close:
    #    expected_vned = euler_to_dcm(*euler).T @ vb

 
    initial_state = SixDOFState(
        Xned=xned, 
        Vb=vb,
        Euler=euler,
        pqr=pqr,
        Vned=vned, 
    )
 
    
    dynamics = SixDOFDynamics(rigid_body, method="rk4", fixed_step_size=0.01)
    nxt_state, _ = dynamics.run_simulation(initial_state, forces, moments, 0.01)
    nxt_xned = nxt_state.Xned
    nxt_vb = nxt_state.Vb
    nxt_euler = nxt_state.Euler
    nxt_pqr = nxt_state.pqr
    nxt_vned = nxt_state.Vned
    
    
    print(f"Vned Expected: {expected_vned}, Got: {nxt_vned}, Close: {jnp.allclose(expected_vned, nxt_vned, atol=threshold)}, Max Error: {jnp.max(jnp.abs(expected_vned - nxt_vned))}")
    print(f"Xned Expected: {expected_xned}, Got: {nxt_xned}, Close: {jnp.allclose(expected_xned, nxt_xned, atol=threshold)}, Max Error: {jnp.max(jnp.abs(expected_xned - nxt_xned))}")
    print(f"Vb Expected: {expected_vb}, Got: {nxt_vb}, Close: {jnp.allclose(expected_vb, nxt_vb, atol=threshold)}, Max Error: {jnp.max(jnp.abs(expected_vb - nxt_vb))}")
    print(f"Euler Expected: {expected_euler}, Got: {nxt_euler}, Close: {jnp.allclose(expected_euler, nxt_euler, atol=threshold)}, Max Error: {jnp.max(jnp.abs(expected_euler - nxt_euler))}")
    print(f"pqr Expected: {expected_pqr}, Got: {nxt_pqr}, Close: {jnp.allclose(expected_pqr, nxt_pqr, atol=threshold)}, Max Error: {jnp.max(jnp.abs(expected_pqr - nxt_pqr))}")
     