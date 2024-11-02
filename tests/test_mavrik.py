import jax.numpy as jnp
from jax import random
import pytest
from jax_mavrik.mavrik import Mavrik

@pytest.fixture
def initial_conditions():
    t = jnp.array([0.0])
    U = 30.0  # trim speed
    eulerIn = jnp.array([0, 4 * jnp.pi / 180, 0])  # trim attitude
    vnedIn = jnp.array([U * jnp.cos(eulerIn[1]), U * jnp.sin(eulerIn[1]), 0])  # NED velocity
    pqrIn = jnp.array([0, 0, 0])  # trim rates
    return t, U, eulerIn, vnedIn, pqrIn

@pytest.fixture
def initial_actuator_settings():
    actuatorsIn = jnp.array([
        0, 0, 0, 0, 0, 0, 0,  # wing_tilt, tail_tilt, aileron, elevator, flap, rudder
        7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500,  # RPM settings
        7500, 7500, 7500, 7500, 7500, 7500
    ])
    return actuatorsIn

def test_initial_conditions(initial_conditions):
    t, U, eulerIn, vnedIn, pqrIn = initial_conditions

    assert U == 30.0
    assert jnp.allclose(eulerIn, jnp.array([0, 4 * jnp.pi / 180, 0]))
    assert jnp.allclose(vnedIn, jnp.array([U * jnp.cos(4 * jnp.pi / 180), U * jnp.sin(4 * jnp.pi / 180), 0]))
    assert jnp.allclose(pqrIn, jnp.array([0, 0, 0]))

def test_initial_actuator_settings(initial_actuator_settings):
    actuatorsIn = initial_actuator_settings

    assert jnp.allclose(actuatorsIn[7:], jnp.full(14, 7500))
    assert jnp.allclose(actuatorsIn[:7], jnp.zeros(7))

def test_simulation_output(initial_conditions, initial_actuator_settings):
    t, U, eulerIn, vnedIn, pqrIn = initial_conditions
    actuatorsIn = initial_actuator_settings

    # Run the JAX simulation
    data = Mavrik(t, U, eulerIn, vnedIn, pqrIn, actuatorsIn)

    # Extract Data
    Fx_vec = data['Forces'][:, 0]
    Fy_vec = data['Forces'][:, 1]
    Fz_vec = data['Forces'][:, 2]
    L_vec = data['Moments'][:, 0]
    M_vec = data['Moments'][:, 1]
    N_vec = data['Moments'][:, 2]

    # Verify simulation output
    assert Fx_vec.size > 0
    assert Fy_vec.size > 0
    assert Fz_vec.size > 0
    assert L_vec.size > 0
    assert M_vec.size > 0
    assert N_vec.size > 0
