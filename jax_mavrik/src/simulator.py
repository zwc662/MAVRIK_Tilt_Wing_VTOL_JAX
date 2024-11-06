
import numpy as np
import jax.numpy as jnp

from jaxtyping import Array, Float
from jax_mavrik.mavrik_types import StateVariables, ControlInputs
from jax_mavrik.src.sixdof import (
    SixDOFDynamics, 
    RigidBody, 
    State as SixDOFState
) 

from jax_mavrik.src.mavrik_aero import MavrikAero
from jax_mavrik.mavrik_setup import MavrikSetup

class Simulator:
    def __init__(self, mass: Float, inertia: Float, mavrik_setup: MavrikSetup):
        rigid_body = RigidBody(mass=mass, inertia=inertia)
        self.sixdof_model = SixDOFDynamics(rigid_body)
        self.aero_model = MavrikAero(mass, mavrik_setup)
 

    def __call__(self, state: StateVariables, control: ControlInputs, dt: Float) -> StateVariables:
        # Calculate forces and moments using Mavrik Aero model
        forces, moments = self.aero_model(state, control)
        
        sixdof_state = SixDOFState(
            position=jnp.array([state.X, state.Y, state.Z]),
            velocity=jnp.array([state.Vx, state.Vy, state.Vz]),
            euler_angles=jnp.array([state.roll, state.pitch, state.yaw]),
            angular_velocity=jnp.array([state.wx, state.wy, state.wz])
        )
        # Compute the state derivatives using 6DOF dynamics
        results_rk4 = self.sixdof_model.run_simulation(sixdof_state, forces, moments, 0, dt, method="RK4")
        # Plot results for RK4 method (position over time as an example)
        nxt_sixdof_state = results_rk4["states"]
        
        return state.replace(
            Vx = nxt_sixdof_state[0],
            Vy = nxt_sixdof_state[1],
            Vz = nxt_sixdof_state[2],
            X = nxt_sixdof_state[3],
            Y = nxt_sixdof_state[4],
            Z = nxt_sixdof_state[5],
            roll = nxt_sixdof_state[6],
            pitch = nxt_sixdof_state[7],
            yaw = nxt_sixdof_state[8],
            wx = nxt_sixdof_state[9],
            wy = nxt_sixdof_state[10],
            wz = nxt_sixdof_state[11],
        )