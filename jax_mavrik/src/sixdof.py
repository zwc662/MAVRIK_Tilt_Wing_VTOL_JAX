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
    velocity: FloatScalar
    position: FloatScalar
    euler_angles: FloatScalar
    angular_velocity: FloatScalar

class SixDOFDynamics:
    """
    Class to simulate 6 Degrees of Freedom (6DOF) dynamics using Euler angles, following
    the behavior of the MathWorks 6DOF block.
    """

    def __init__(self, rigid_body: RigidBody, method: str = 'RK4', fixed_step_size: float = 0.01):
        """
        Initialize the 6DOF dynamics simulator.

        Args:
            rigid_body (RigidBody): Rigid body object containing mass and inertia.
            method (str): Integration method ("RK4", "Euler", or "diffrax"). 
            fixd_step_size (float): Fixed step size for the simulation.
        """
        self.rigid_body = rigid_body 
        self.method = method 
        self.fixed_step_size = fixed_step_size
        
    def _six_dof_dynamics(self, state, Fxyz, Mxyz):
        """
        Defines the 6DOF dynamics equations of motion based on Newton's and Euler's equations.
        
        Args:
            state (numpy.ndarray): Current state vector [u, v, w, phi, theta, psi, p, q, r].
            Fxyz (array-like): Forces acting on the rigid body (Fx, Fy, Fz).
            Mxyz (array-like): Moments acting on the rigid body (Mx, My, Mz).
        
        Returns:
            numpy.ndarray: Derivative of the state vector.
        """
         
        array, cos, sin, tan, concatenate = np.asarray, np.cos, np.sin, np.tan, np.concatenate
        if isinstance(state, jnp.ndarray):
            array, cos, sin, tan, concatenate = jnp.asarray, jnp.cos, jnp.sin, jnp.tan, jnp.concatenate
          
        u, v, w, x, y, z, phi, theta, psi, p, q, r = state

        # Convert body-frame forces (Fxyz) to NED-frame forces
        R_body_to_ned = self._euler_to_dcm(array([phi, theta, psi])).T  # Transpose to convert body to NED
        F_ned = R_body_to_ned @ Fxyz  # Transform forces to NED frame

        # Compute acceleration in the NED frame
        V_ned = jnp.array([u, v, w])  # Velocity in NED frame
        force_term_ned = F_ned / self.rigid_body.mass
        angular_velocity = R_body_to_ned @ jnp.array([p, q, r])
        cross_product_term = jnp.cross(angular_velocity, V_ned)  # Coriolis effect in NED frame

        # Total acceleration in the NED frame
        V_ned_dot = force_term_ned - cross_product_term

        # Rotational motion (Euler's equations in the body frame)
        I = self.rigid_body.inertia
        I_inv = jnp.linalg.inv(I)
        Mxyz = jnp.array(Mxyz)
        angular_acceleration = I_inv @ (Mxyz - jnp.cross(angular_velocity, I @ angular_velocity))
        dp, dq, dr = angular_acceleration

        # Euler angles rates
        dphi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
         
        # Position and velocity in the NED frame (already in NED frame, no additional transformation needed)
        du = V_ned_dot[0] 
        dv = V_ned_dot[1] 
        dw = V_ned_dot[2] 
         
        velocity_xyz = R_body_to_ned @ V_ned  # Convert NED velocities to XYZ frame for position update
        dx = velocity_xyz[0]
        dy = velocity_xyz[1]
        dz = velocity_xyz[2]

        # Return the state offsets
        return concatenate([array([du, dv, dw]), array([dx, dy, dz]), array([dphi, dtheta, dpsi]), array([dp, dq, dr])])
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
        
        # phi, theta, psi = euler_angles
        roll, pitch, yaw = euler_angles
        cos, sin = np.cos, np.sin
        array = np.asarray
        if isinstance(euler_angles, jnp.ndarray):
            array, cos, sin = jnp.asarray, jnp.cos, jnp.sin
          
        s_roll = sin(roll) #1
        s_pitch = sin(pitch) #2
        s_yaw = sin(yaw) #3

        c_roll = cos(roll) #4
        c_pitch = cos(pitch) #5
        c_yaw = cos(yaw) #6

        # Rotation matrix from body frame to inertial frame
        return array([
            [c_pitch * c_yaw, c_pitch * s_yaw, -s_pitch],
            [s_roll * s_pitch * c_yaw - c_roll * s_yaw, s_roll * s_pitch * s_yaw + c_roll * c_yaw, s_roll * c_pitch],
            [c_roll * s_pitch * c_yaw + s_roll * s_yaw, c_roll * s_pitch * s_yaw - s_roll * c_yaw, c_roll * c_pitch]
        ])
 
    def run_simulation(self, initial_state: State, forces: FloatScalar, moments: FloatScalar, t0=0.0, t1=0.01):
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
        initial_state_vector = jnp.concatenate([
            initial_state.velocity,
            initial_state.position,
            initial_state.euler_angles, 
            initial_state.angular_velocity
        ])
        
        forces = jnp.asarray(forces)
        moments = jnp.asarray(moments)

        num_points = jnp.ceil((t1 - t0) / self.fixed_step_size).astype(int)
        times = jnp.linspace(t0, t1, num_points)
        if self.method.lower() == "rk4":
            def rk4_step(state, forces_moments):
                forces, moments = forces_moments
                forces, moments = jnp.asarray(forces), jnp.asarray(moments)
                k1 = self.fixed_step_size * self._six_dof_dynamics(state, forces, moments)
                k2 = self.fixed_step_size * self._six_dof_dynamics(state + k1/2, forces, moments)
                k3 = self.fixed_step_size * self._six_dof_dynamics(state + k2/2, forces, moments)
                k4 = self.fixed_step_size * self._six_dof_dynamics(state + k3, forces, moments)
                new_state = state + (k1 + 2*k2 + 2*k3 + k4) / 6
                return new_state, new_state

            forces_moments = (jnp.tile(forces, (num_points, 1)), jnp.tile(moments, (num_points, 1)))
            _, states = lax.scan(rk4_step, initial_state_vector, forces_moments)

        elif self.method.lower() == "euler":
            def euler_step(state, forces_moments):
                forces, moments = forces_moments
                forces, moments = jnp.asarray(forces), jnp.asarray(moments)
                new_state = state + self.fixed_step_size * self._six_dof_dynamics(state, forces, moments)
                return new_state, new_state

            forces_moments = (jnp.tile(forces, (num_points, 1)), jnp.tile(moments, (num_points, 1)))
            _, states = lax.scan(euler_step, initial_state_vector, forces_moments)
        elif self.method.lower() == "diffrax":
            def dynamics(t, y, args):
                forces, moments = jnp.asarray(args[0]), jnp.asarray(args[1])
                return jnp.array(self._six_dof_dynamics(y, forces, moments))

            solver = diffrax.Tsit5()
            term = diffrax.ODETerm(dynamics)
            saveat = diffrax.SaveAt(ts=times)
            sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=self.fixed_step_size, y0=initial_state_vector, args=(forces, moments), saveat=saveat)
            states = jnp.array(sol.ys)

        return {"time": times, "states": states} # Return time and state history

if __name__ == "__main__":
    # Define constants and initial state
    mass = 10.0
    inertia = np.diag([0.5, 0.5, 0.8])
    forces = [0., 0., -mass * 9.81]  # Gravity in the body frame
    moments = [0., 0., 0.]  # No initial moments
    t0, t1 = 0.0, 10.0
    initial_state = State(
        position=np.array([0., 0., 0.]),
        velocity=np.array([30., 0., 0.]),
        euler_angles=np.array([0., 0., 0.]),
        angular_velocity=np.array([0., 0., 0.])
    )

    rigid_body = RigidBody(mass=mass, inertia=np.array(inertia))

    plt.figure()
    
    ## Testing RK4 method
    dynamics = SixDOFDynamics(rigid_body, method="RK4", fixed_step_size=0.01)
    results_rk4 = dynamics.run_simulation(initial_state, forces, moments, t0, t1)
    # Plot results for RK4 method (position over time as an example)
    time_rk4 = results_rk4["time"]
    position_rk4 = results_rk4["states"][:, :3]  # x, y, z positions
    plt.plot(time_rk4, position_rk4[:, 3], label="X Position (RK4)")
    plt.plot(time_rk4, position_rk4[:, 4], label="Y Position (RK4)")
    plt.plot(time_rk4, position_rk4[:, 5], label="Z Position (RK4)")

    ## Testing Euler method
    dynamics = SixDOFDynamics(rigid_body, method="euler", fixed_step_size=0.01)
    results_euler = dynamics.run_simulation(initial_state, forces, moments, t0, t1)
    time_euler = results_euler["time"]
    position_euler = results_euler["states"][:, :3]  # x, y, z positions
    plt.plot(time_euler, position_euler[:, 3], label="X Position (Euler)")
    plt.plot(time_euler, position_euler[:, 4], label="Y Position (Euler)")
    plt.plot(time_euler, position_euler[:, 5], label="Z Position (Euler)")

    ## Testing diffrax method
    dynamics = SixDOFDynamics(rigid_body, method="diffrax", fixed_step_size=0.01)
    results_diffrax = dynamics.run_simulation(initial_state, forces, moments, t0, t1)
    time_diffrax = results_diffrax["time"]
    position_diffrax = results_diffrax["states"][:, :3]  # x, y, z positions
    plt.plot(time_diffrax, position_diffrax[:, 3], label="X Position (diffrax)")
    plt.plot(time_diffrax, position_diffrax[:, 4], label="Y Position (diffrax)")
    plt.plot(time_diffrax, position_diffrax[:, 5], label="Z Position (diffrax)")


    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time (All Methods)")
    plt.legend()
    plt.show()
