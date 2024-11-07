
import numpy as np
import jax.numpy as jnp

from jaxtyping import Array, Float
from jax_mavrik.mavrik_types import StateVariables, ControlInputs
from jax_mavrik.src.sixdof import (
    SixDOFDynamics, 
    RigidBody, 
    State as SixDOFState
) 
from jax import jit

from jax_mavrik.src.mavrik_aero import MavrikAero
from jax_mavrik.mavrik_setup import MavrikSetup

class Simulator:
    def __init__(self, mass: Float, inertia: Float, mavrik_setup: MavrikSetup):
        rigid_body = RigidBody(mass=mass, inertia=inertia)
        self.sixdof_model = SixDOFDynamics(rigid_body)
        self.aero_model = MavrikAero(mass, mavrik_setup)
 

    def run(self, state: StateVariables, control: ControlInputs, dt: Float) -> StateVariables:
        # Calculate forces and moments using Mavrik Aero model
        forces, moments = self.aero_model(state, control) 
        sixdof_state = SixDOFState(
            position=jnp.array([state.X, state.Y, state.Z]),
            velocity=jnp.array([state.Vx, state.Vy, state.Vz]),
            euler_angles=jnp.array([state.roll, state.pitch, state.yaw]),
            angular_velocity=jnp.array([state.wx, state.wy, state.wz])
        )
        sixdof_forces = jnp.array([forces.Fx, forces.Fy, forces.Fz])
        sixdof_moments = jnp.array([moments.L, moments.M, moments.N])
        # Compute the state derivatives using 6DOF dynamics
        results_rk4 = self.sixdof_model.run_simulation(sixdof_state, sixdof_forces, sixdof_moments, 0, dt, method="RK4")
        # Plot results for RK4 method (position over time as an example)
        nxt_sixdof_state = results_rk4["states"][-1]
        
        return state._replace(
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

if __name__ == "__main__":
    # Initialize MavrikSetup with appropriate values
    mavrik_setup = MavrikSetup(file_path="/Users/weichaozhou/Workspace/Mavrik_JAX/jax_mavrik/aero_export.mat")

    # Define constants
    mass = 10.0
    inertia = [0.5, 0.5, 0.8]
    dt = 0.01  # Time step

    # Initialize Simulator
    simulator = Simulator(mass=mass, inertia=inertia, mavrik_setup=mavrik_setup)

    # Define initial state variables
    state = StateVariables(
        Vx=10.0, Vy=0.0, Vz=0.0,
        X=0.0, Y=0.0, Z=0.0,
        roll=0.0, pitch=0.0, yaw=0.0,
        Vbx=0.0, Vby=0.0, Vbz=0.0,
        wx=0.0, wy=0.0, wz=0.0,
        dwdt_x=0.0, dwdt_y=0.0, dwdt_z=0.0,
        ax=0.0, ay=0.0, az=0.0,
        Fx=0.0, Fy=0.0, Fz=0.0,
        L=0.0, M=0.0, N=0.0
    )

    # Define control inputs
    control = ControlInputs(
        wing_tilt=0.0, tail_tilt=0.0, aileron=0.0,
        elevator=0.0, flap=0.0, rudder=0.0,
        RPM_tailLeft=1000.0, RPM_tailRight=1000.0,
        RPM_leftOut1=1000.0, RPM_left2=1000.0,
        RPM_left3=1000.0, RPM_left4=1000.0,
        RPM_left5=1000.0, RPM_left6In=1000.0,
        RPM_right7In=1000.0, RPM_right8=1000.0,
        RPM_right9=1000.0, RPM_right10=1000.0,
        RPM_right11=1000.0, RPM_right12Out=1000.0
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
    plt.plot(times, positions[:, 0], label="X Position (RK4)")
    plt.plot(times, positions[:, 1], label="Y Position (RK4)")
    plt.plot(times, positions[:, 2], label="Z Position (RK4)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time (RK4)")
    plt.legend()
    plt.show()
     