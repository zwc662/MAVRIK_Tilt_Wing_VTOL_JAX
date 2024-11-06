from jax_mavrik.mavrik_types import State, Control 

# mavrik.py
from jax_mavrik.src.simulator import Simulator
from jax_mavrik.mavrik_setup import MavrikSetup
from jax_mavrik.mavrik_types import ControlInputs, State
from jax_mavrik.mavrik_types import StateVariables, ControlInputs

import numpy as np 
from typing import Dict, Any, Tuple
from diffrax import ODETerm, Tsit5, diffeqsolve

import jax.numpy as jnp 


class Mavrik:
    def __init__(self, mass: float, inertia: Tuple[float, float, float]):
        mavrik_setup = MavrikSetup()

        self.sim = Simulator(mass = mass, inertia = inertia, mavrik_setup = mavrik_setup)

    def step(self, dt: float, state: StateVariables, control: ControlInputs) -> jnp.ndarray:
        # Calculate forces and moments using Mavrik Aero model
        nxt_state: StateVariables = self.sim(state, control, control, dt)
        # Compute the state derivatives using 6DOF dynamics
        return nxt_state
 