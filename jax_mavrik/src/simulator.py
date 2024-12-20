
import numpy as np
import jax.numpy as jnp

from jaxtyping import Array, Float
from jax_mavrik.mavrik_types import StateVariables, ControlInputs
from jax_mavrik.src.sixdof import (
    SixDOFDynamics, 
    RigidBody, 
    SixDOFState
) 
from jax import jit

from jax_mavrik.src.mavrik_aero import MavrikAero
from jax_mavrik.mavrik_setup import MavrikSetup

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Simulator:
    def __init__(self, mavrik_setup: MavrikSetup, method: str = 'diffrax', fixed_step_size: float = 0.01):
        rigid_body = RigidBody(mass=mavrik_setup.mass, inertia=mavrik_setup.inertia, inverse_inertia=np.linalg.inv(mavrik_setup.inertia))
        self.sixdof_model = SixDOFDynamics(rigid_body, method, fixed_step_size) 
        self.aero_model = MavrikAero(mavrik_setup)
 

    def run(self, state: StateVariables, control: ControlInputs, dt: Float) -> StateVariables:
        # Calculate forces and moments using Mavrik Aero model
        forces, moments, _ = self.aero_model(state, control) 
        sixdof_state = SixDOFState(
            Vned=np.array([state.VXe, state.VYe, state.VZe]),
            Xned=np.array([state.Xe, state.Ye, state.Ze]),
            Vb=np.array([state.u, state.v, state.w]), 
            Euler=wrap_to_pi(np.array([state.roll, state.pitch, state.yaw])),
            pqr=np.array([state.p, state.q, state.r])
        )
 
        sixdof_forces = np.array([forces.Fx, forces.Fy, forces.Fz])
        sixdof_moments = np.array([moments.L, moments.M, moments.N])
        # Compute the state derivatives using 6DOF dynamics
        nxt_sixdof_state, info = self.sixdof_model.run_simulation(sixdof_state, sixdof_forces, sixdof_moments, dt) 
        
        # Ensure angles are within [-np.pi, np.pi]
        
        nxt_state = state._replace(
            VXe=nxt_sixdof_state.Vned[0],
            VYe=nxt_sixdof_state.Vned[1],
            VZe=nxt_sixdof_state.Vned[2],
            Xe=nxt_sixdof_state.Xned[0],
            Ye=nxt_sixdof_state.Xned[1],
            Ze=nxt_sixdof_state.Xned[2],
            u=nxt_sixdof_state.Vb[0],
            v=nxt_sixdof_state.Vb[1],
            w=nxt_sixdof_state.Vb[2],
            roll=wrap_to_pi(nxt_sixdof_state.Euler[0]),
            pitch=wrap_to_pi(nxt_sixdof_state.Euler[1]),
            yaw=wrap_to_pi(nxt_sixdof_state.Euler[2]),
            p=nxt_sixdof_state.pqr[0],
            q=nxt_sixdof_state.pqr[1],
            r=nxt_sixdof_state.pqr[2],
            Fx=forces.Fx,
            Fy=forces.Fy,
            Fz=forces.Fz,
            L=moments.L,
            M=moments.M,
            N=moments.N
        )
        return nxt_state, info

if __name__ == "__main__":
    # Initialize MavrikSetup with appropriate values
    mavrik_setup = MavrikSetup(file_path="/Users/weichaozhou/Workspace/Mavrik_JAX/jax_mavrik/aero_export.mat")

    # Define constants
    dt = 0.1  # Time step

    # Initialize Simulator
    simulator = Simulator(mavrik_setup=mavrik_setup)

    # Define initial state variables
    state = StateVariables(
        u=29.9269, v=0.0, w=2.0927,
        Xe=0.0, Ye=0.0, Ze=0.0,
        roll=0.0, pitch=0.069813, yaw=0.0,
        VXe=30.0, VYe=0.0, VZe=0.0,
        p=0.0, q=0.0, r=0.0,
        Fx=0.0, Fy=0.0, Fz=0.0,
        L=0.0, M=0.0, N=0.0
    )

    # Define control inputs
    control = ControlInputs(
        wing_tilt=0.0, tail_tilt=0.0, aileron=0.0,
        elevator=0.0, flap=0.0, rudder=0.0,
        RPM_tailLeft=7500, RPM_tailRight=7500,
        RPM_leftOut1=7500, RPM_left2=7500,
        RPM_left3=7500, RPM_left4=7500,
        RPM_left5=7500, RPM_left6In=7500,
        RPM_right7In=7500, RPM_right8=7500,
        RPM_right9=7500, RPM_right10=7500,
        RPM_right11=7500, RPM_right12Out=7500
    )

    # Run the simulation for a certain number of steps
    num_steps = 10
    states = [state]
    times = np.linspace(0, dt * num_steps, num_steps)
    for _ in range(num_steps):
        state = simulator.run(state, control, dt)
        states.append(state)

    # Print the final state
    print("States:", states)

    import matplotlib.pyplot as plt
    positions = np.array([state[:3] for state in states[:-1]])
    plt.figure()
    plt.plot(times, positions[:, 3], label="X Position (RK4)")
    plt.plot(times, positions[:, 4], label="Y Position (RK4)")
    plt.plot(times, positions[:, 5], label="Z Position (RK4)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time (RK4)")
    plt.legend()
    plt.show()
     