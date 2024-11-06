from jax_mavrik.mavrik_types import FULL_STATE_NX, OUTER_CONTROL_NU, FullState, OuterControl

# mavrik.py

from jax_mavrik.mavrik_types import ControlInputs, State
from jax_mavrik.src.sixdof import SixDOFDynamics
from jax_mavrik.src.mavrik_aero import MavrikAero
 
from typing import Dict, Any, Tuple
from diffrax import ODETerm, Tsit5, diffeqsolve

class Mavrik:
    def __init__(self, mass: float, inertia: Tuple[float, float, float], initial_state: State):
        self.sixdof = SixDOFDynamics(mass, inertia)
        self.aero_model = MavrikAero(mass)
        self.state = jnp.array([
            *initial_state.position,
            *initial_state.velocity,
            *initial_state.orientation,
            *initial_state.angular_rates
        ])

    def _dynamics(self, t: float, state: jnp.ndarray, control_inputs: ControlInputs) -> jnp.ndarray:
        # Calculate forces and moments using Mavrik Aero model
        forces, moments = self.aero_model.calculate_forces_and_moments(control_inputs, state)
        
        # Compute the state derivatives using 6DOF dynamics
        return self.sixdof.calculate_dynamics(state, forces, moments)

    def step(self, control_inputs: ControlInputs, dt: float = 0.01) -> Dict[str, Any]:
        """
        Perform a continuous-time simulation step.

        Args:
            control_inputs: ControlInputs object with control variables.
            dt: Simulation time step.

        Returns:
            Updated state as a dictionary.
        """
        term = ODETerm(self._dynamics)
        solution = diffeqsolve(
            term,
            solver=Tsit5(),
            t0=0.0,
            t1=dt,
            dt0=0.01,
            y0=self.state,
            args=(control_inputs,)
        )
        self.state = solution.ys[-1]
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """
        Retrieve the current state.

        Returns:
            Dictionary containing position, velocity, orientation, and angular rates.
        """
        return {
            "position": self.state[:3],
            "velocity": self.state[3:6],
            "orientation": self.state[6:9],
            "angular_rates": self.state[9:12]
        }