import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from jax_mavrik.mavrik import Mavrik

import jax.numpy as jnp

import numpy as np
import datetime

import os
import sys
import pickle




class Trim():
    def __init__(self):
        self.mavrik = Mavrik()

    def search_trimming_state(self, initial_state: jnp.ndarray, initial_control: jnp.ndarray, 
                                tolerance: float = 1e-6, max_iterations: int = 1000) -> Optional[tuple]:
        state = initial_state
        control = initial_control

        for iteration in range(max_iterations):
            state_derivative = self.mavrik.dynamics(state, control)
            error = jnp.linalg.norm(state_derivative)

            if error < tolerance:
                return state, control

            # Update control using some optimization method
            control = self.update_control(state, control, state_derivative)

        return None

    def update_control(self, state: jnp.ndarray, control: jnp.ndarray, state_derivative: jnp.ndarray) -> jnp.ndarray:
        # Placeholder for control update logic
        # This could be a gradient descent step, a PID controller, or any other method
        # For now, we just return the control unchanged
        return control