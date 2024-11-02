# mavrik_aero.py

from mavrik_types import ControlInputs
import jax.numpy as jnp
from typing import Tuple

class MavrikAero:
    def __init__(self, mass: float):
        self.mass = mass

    def calculate_forces_and_moments(self, control_inputs: ControlInputs, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Calculate forces and moments based on control inputs and current state.

        Args:
            control_inputs: ControlInputs with tilt angles, RPMs, and control surfaces.
            state: Current state vector.

        Returns:
            Tuple of forces and moments as jnp.ndarray.
        """
        # Placeholder calculations
        Fx = jnp.sum(jnp.array(control_inputs.rpms)) * 0.1  # Simple thrust based on RPM
        Fz = Fx * jnp.sin(control_inputs.tilt_angles[0])  # Adjust thrust direction by tilt angle
        forces = jnp.array([Fx, 0, Fz])

        moments = jnp.array([
            control_inputs.control_surfaces[0] * 0.05,
            control_inputs.control_surfaces[1] * 0.05,
            control_inputs.control_surfaces[2] * 0.05
        ])

        return forces, moments