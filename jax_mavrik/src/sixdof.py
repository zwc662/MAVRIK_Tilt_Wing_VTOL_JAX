
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve 
import jax

class RigidBody:
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = jnp.array(inertia)

class State:
    def __init__(self, x, y, z, u, v, w, phi, theta, psi, p, q, r):
        self.position = jnp.array([x, y, z])
        self.velocity = jnp.array([u, v, w])
        self.euler_angles = jnp.array([phi, theta, psi])
        self.angular_velocity = jnp.array([p, q, r])

class SixDOFDynamics:
    """
    Class to simulate 6 Degrees of Freedom (6DOF) dynamics using Euler angles, following
    the behavior of the MathWorks 6DOF block.
    """

    def __init__(self, rigid_body):
        """
        Initialize the 6DOF dynamics simulator.

        Args:
            rigid_body (RigidBody): Rigid body object containing mass and inertia.
        """
        self.rigid_body = rigid_body
        self.ode_term = ODETerm(self._six_dof_dynamics)

    def _six_dof_dynamics(self, t, state, args):
        """
        Defines the 6DOF dynamics equations of motion based on Newton's and Euler's equations.
        
        Args:
            t (float): Time (not used in this system but required by diffrax).
            state (jax.numpy.ndarray): Current state vector [x, y, z, u, v, w, phi, theta, psi, p, q, r].
            args (tuple): Forces and moments acting on the rigid body (F_xyz, M_xyz).
        
        Returns:
            jax.numpy.ndarray: Derivative of the state vector.
        """
        Fxyz, Mxyz = args
        x, y, z, u, v, w, phi, theta, psi, p, q, r = state

        # Translational motion (Newton's second law in the body frame)
        ax = Fxyz[0] / self.rigid_body.mass
        ay = Fxyz[1] / self.rigid_body.mass
        az = Fxyz[2] / self.rigid_body.mass

        # Compute body-frame linear accelerations
        Vb_dot = jnp.array([ax, ay, az])

        # Rotational motion (Euler's equations in the body frame)
        Ix, Iy, Iz = self.rigid_body.inertia
        dp = (Mxyz[0] - (Iy - Iz) * q * r) / Ix
        dq = (Mxyz[1] - (Iz - Ix) * p * r) / Iy
        dr = (Mxyz[2] - (Ix - Iy) * p * q) / Iz

        # Euler angles rates
        dphi = p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta)
        dtheta = q * jnp.cos(phi) - r * jnp.sin(phi)
        dpsi = q * jnp.sin(phi) / jnp.cos(theta) + r * jnp.cos(phi) / jnp.cos(theta)

        # Position and velocity in the NED frame
        R = self._euler_to_dcm(phi, theta, psi)
        V_ned = R @ jnp.array([u, v, w])

        # Return the derivative of the state vector
        return jnp.concatenate([V_ned, Vb_dot, jnp.array([dphi, dtheta, dpsi]), jnp.array([dp, dq, dr])])

    def _euler_to_dcm(self, phi, theta, psi):
        """
        Calculates the Direction Cosine Matrix (DCM) from Euler angles.
        
        Args:
            phi (float): Roll angle.
            theta (float): Pitch angle.
            psi (float): Yaw angle.
        
        Returns:
            jax.numpy.ndarray: The 3x3 Direction Cosine Matrix (DCM).
        """
        cos, sin = jnp.cos, jnp.sin
        return jnp.array([
            [cos(theta) * cos(psi), cos(theta) * sin(psi), -sin(theta)],
            [sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi),
             sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
             sin(phi) * cos(theta)],
            [cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi),
             cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
             cos(phi) * cos(theta)]
        ])

    def run_simulation(self, initial_state, forces, moments, t0=0.0, t1=10.0, num_points=100):
        """
        Run the 6DOF dynamics simulation.
        
        Args:
            initial_state (State): Initial state object.
            forces (array-like): External forces in the body frame (Fx, Fy, Fz).
            moments (array-like): External moments in the body frame (Mx, My, Mz).
            t0 (float): Initial time of the simulation.
            t1 (float): Final time of the simulation.
            num_points (int): Number of points for evaluation.
        
        Returns:
            dict: A dictionary containing time and state history.
        """
        initial_state_vector = jnp.concatenate([
            initial_state.position,
            initial_state.velocity,
            initial_state.euler_angles,
            initial_state.angular_velocity
        ])
        
        saveat = SaveAt(ts=jnp.linspace(t0, t1, num_points))
        
        # Solve the ODE
        solution = diffeqsolve(
            self.ode_term,
            solver=Tsit5(),
            t0=t0,
            t1=t1,
            dt0=0.1,
            y0=initial_state_vector,
            args=(forces, moments),
            saveat=saveat
        )

        return {"time": solution.ts, "states": solution.ys}
        # Example Usage

if __name__ == "__main__":
    # Define constants and initial state
    mass = 10.0
    inertia = [0.5, 0.5, 0.8]
    forces = [0, 0, -mass * 9.81]  # Gravity in the body frame
    moments = [0, 0, 0]  # No initial moments
    initial_state = State(0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0, 0)

    rigid_body = RigidBody(mass, inertia)
    dynamics = SixDOFDynamics(rigid_body)
    
    results = dynamics.run_simulation(initial_state, forces, moments)

    # Plot results (position over time as an example)
    time = results["time"]
    position = results["states"][:, :3]  # x, y, z positions

    plt.figure()
    plt.plot(time, position[:, 0], label="X Position")
    plt.plot(time, position[:, 1], label="Y Position")
    plt.plot(time, position[:, 2], label="Z Position")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("6DOF Position Over Time")
    plt.legend()
    plt.show()
    # Example Usage
 