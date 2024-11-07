import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple, Optional
from jaxtyping import Float

from jax_mavrik.src.utils.jax_types import FloatScalar
import diffrax
from jax import numpy as jnp
from jax import jit
from jax import lax


class RigidBody(NamedTuple):
    mass: Float
    inertia: Float

class State(NamedTuple):
    position: FloatScalar
    velocity: FloatScalar
    euler_angles: FloatScalar
    angular_velocity: FloatScalar

class SixDOFDynamics:
    """
    Class to simulate 6 Degrees of Freedom (6DOF) dynamics using Euler angles, following
    the behavior of the MathWorks 6DOF block.
    """

    def __init__(self, rigid_body: RigidBody, dt: float = 0.01):
        """
        Initialize the 6DOF dynamics simulator.

        Args:
            rigid_body (RigidBody): Rigid body object containing mass and inertia.
        """
        self.rigid_body = rigid_body
        self.dt = dt

    def _six_dof_dynamics(self, t, state, Fxyz, Mxyz):
        """
        Defines the 6DOF dynamics equations of motion based on Newton's and Euler's equations.
        
        Args:
            t (float): Time (not used in this system but required by diffrax).
            state (numpy.ndarray): Current state vector [x, y, z, u, v, w, phi, theta, psi, p, q, r].
            args (tuple): Forces and moments acting on the rigid body (F_xyz, M_xyz).
        
        Returns:
            numpy.ndarray: Derivative of the state vector.
        """
        
        x, y, z, u, v, w, phi, theta, psi, p, q, r = state

        # Translational motion (Newton's second law in the body frame)
        Vb_dot = Fxyz / self.rigid_body.mass

        # Rotational motion (Euler's equations in the body frame)
        Ix, Iy, Iz = self.rigid_body.inertia
        dp = (Mxyz[0] - (Iy - Iz) * q * r) / Ix
        dq = (Mxyz[1] - (Iz - Ix) * p * r) / Iy
        dr = (Mxyz[2] - (Ix - Iy) * p * q) / Iz
        
        array, cos, sin, tan, concatenate = np.asarray, np.cos, np.sin, np.tan, np.concatenate
        if isinstance(state, jnp.ndarray):
            array, cos, sin, tan, concatenate = jnp.asarray, jnp.cos, jnp.sin, jnp.tan, jnp.concatenate
          
        # Euler angles rates
        dphi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
         
        # Position and velocity in the NED frame
        #R = self._euler_to_dcm(array([phi, theta, psi]))
        #V_ned = R @ array([u, v, w])
        #return concatenate([V_ned, Vb_dot, array([dphi, dtheta, dpsi]), array([dp, dq, dr])])

        Vxyz = array([u, v, w])
        # Return the derivative of the state vector
        return concatenate([Vxyz, Vb_dot, array([dphi, dtheta, dpsi]), array([dp, dq, dr])])

    def _euler_to_dcm(self, euler_angles):
        """
        Calculates the Direction Cosine Matrix (DCM) from Euler angles.
        
        Args:
            phi (float): Roll angle.
            theta (float): Pitch angle.
            psi (float): Yaw angle.
        
        Returns:
            jax.numpy.ndarray: The 3x3 Direction Cosine Matrix (DCM).
        """
        phi, theta, psi = euler_angles
        cos, sin = np.cos, np.sin
        array = np.asarray
        if isinstance(euler_angles, jnp.ndarray):
            array, cos, sin = jnp.asarray, jnp.cos, jnp.sin
          
        return array([
            [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
            [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
             sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
             sin(phi) * cos(theta)],
            [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
             cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
             cos(phi) * cos(theta)]
        ])
 
    def run_simulation(self, initial_state: State, forces: FloatScalar, moments: FloatScalar, t0=0.0, t1=0.01, method="RK4"):
        """
        Run the 6DOF dynamics simulation.
        
        Args:
            initial_state (State): Initial state object.
            forces (array-like): External forces in the body frame (Fx, Fy, Fz).
            moments (array-like): External moments in the body frame (Mx, My, Mz).
            t0 (float): Initial time of the simulation.
            t1 (float): Final time of the simulation.
            num_points (int): Number of points for evaluation.
            method (str): Integration method ("RK4" or "diffrax").
        
        Returns:
            dict: A dictionary containing time and state history.
        """
        initial_state_vector = np.concatenate([
            initial_state.position,
            initial_state.velocity,
            initial_state.euler_angles, 
            initial_state.angular_velocity
        ])
        
        num_points = jnp.ceil((t1 - t0) / self.dt).astype(int)
        time = np.linspace(t0, t1, num_points)
        if method == "RK4":
            states = np.zeros((num_points, len(initial_state_vector)))
            states[0] = initial_state_vector
            def rk4_step(state, t_forces_moments):
                t, forces, moments = t_forces_moments
                forces, moments = jnp.asarray(forces), jnp.asarray(moments)
                k1 = self.dt * self._six_dof_dynamics(t, state, forces, moments)
                k2 = self.dt * self._six_dof_dynamics(t + self.dt/2, state + k1/2, forces, moments)
                k3 = self.dt * self._six_dof_dynamics(t + self.dt/2, state + k2/2, forces, moments)
                k4 = self.dt * self._six_dof_dynamics(t + self.dt, state + k3, forces, moments)
                new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
                return new_state, new_state

            t_forces_moments = (time, jnp.tile(forces, (num_points, 1)), jnp.tile(moments, (num_points, 1)))
            _, states = lax.scan(rk4_step, initial_state_vector, t_forces_moments)

            return {"time": time, "states": states} # Return time and state history
        elif method == "diffrax":
            states = jnp.zeros((num_points, len(initial_state_vector)))
            states = states.at[0].set(initial_state_vector)
            def solve_dynamics(initial_state_vector, forces, moments, time):
                def dynamics(t, y, args):
                    forces, moments = jnp.asarray(args[0]), jnp.asarray(args[1])
                    return jnp.array(self._six_dof_dynamics(t, y, forces, moments))

                solver = diffrax.Tsit5()
                term = diffrax.ODETerm(dynamics)
                saveat = diffrax.SaveAt(ts=time)
                sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt, y0=initial_state_vector, args=(forces, moments), saveat=saveat)
                return jnp.array(sol.ys)

            states = solve_dynamics(initial_state_vector, forces, moments, time)

        return {"time": time, "states": states}

if __name__ == "__main__":
    # Define constants and initial state
    mass = 10.0
    inertia = [0.5, 0.5, 0.8]
    forces = [0, 0, -mass * 9.81]  # Gravity in the body frame
    moments = [0, 0, 0]  # No initial moments
    t0, t1 = 0.0, 100.0
    initial_state = State(
        position=np.array([0, 0, 0]),
        velocity=np.array([30, 0, 0]),
        euler_angles=np.array([0, 0, 0]),
        angular_velocity=np.array([0, 0, 0])
    )
    
    rigid_body = RigidBody(mass=mass, inertia=np.array(inertia))
    dynamics = SixDOFDynamics(rigid_body)
    results_rk4 = dynamics.run_simulation(initial_state, forces, moments, t0, t1, method="RK4")
    # Plot results for RK4 method (position over time as an example)
    time_rk4 = results_rk4["time"]
    position_rk4 = results_rk4["states"][:, :3]  # x, y, z positions

    plt.figure()
    plt.plot(time_rk4, position_rk4[:, 0], label="X Position (RK4)")
    plt.plot(time_rk4, position_rk4[:, 1], label="Y Position (RK4)")
    plt.plot(time_rk4, position_rk4[:, 2], label="Z Position (RK4)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time (RK4)")
    plt.legend()
    plt.show()
     

    results_diffrax = dynamics.run_simulation(initial_state, forces, moments, t0, t1, method="diffrax")
    # Plot results for diffrax method (position over time as an example)
    time_diffrax = results_diffrax["time"]
    position_diffrax = results_diffrax["states"][:, :3]  # x, y, z positions

    plt.figure()
    plt.plot(time_diffrax, position_diffrax[:, 0], label="X Position (diffrax)")
    plt.plot(time_diffrax, position_diffrax[:, 1], label="Y Position (diffrax)")
    plt.plot(time_diffrax, position_diffrax[:, 2], label="Z Position (diffrax)")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time (diffrax)")
    plt.legend()
    plt.show()
