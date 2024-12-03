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

from jax_mavrik.examples.system_id import SystemID
import jax
from jax.lib import xla_bridge
jax.config.update('jax_platform_name', 'cpu')


current_dir = os.path.dirname(os.path.abspath(__file__))

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class PIDController:
    def __init__(self, dt, Kp, Ki, Kd):
        self.dt = dt
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

class MavrikPIDController:
    def __init__(self, mavrik: Mavrik, dt: float = 0.01, pos_K: Optional[np.ndarray] = None, att_K: Optional[np.ndarray] = None):
        self.mavrik = mavrik
         
        if pos_K is None:
            pos_K = np.array([[1.0, 0.1, 0.5], [1.0, 0.1, 0.5], [1.5, 0.2, 0.6]])
        if att_K is None:
            att_K = np.array([[1.2, 0.1, 0.4], [1.2, 0.1, 0.4], [1.5, 0.2, 0.5]])

        # PID controllers for ned xned (x, y, z)
        self.pos_pid_x = PIDController(dt, *pos_K[0])
        self.pos_pid_y = PIDController(dt, *pos_K[1])
        self.pos_pid_z = PIDController(dt, *pos_K[2])

        # PID controllers for orientation (roll, pitch, yaw)
        self.att_pid_roll = PIDController(dt, *att_K[0])
        self.att_pid_pitch = PIDController(dt, *att_K[1])
        self.att_pid_yaw = PIDController(dt, *att_K[2])

        # Define expected thrust ranges for normalization
        self.thrust_xyz_min = np.array([-10.0, -10.0, 0.0])  # Example minimum thrust (negative for upward acceleration)
        self.thrust_xyz_max = np.array([10.0, 10.0, 10.0])   # Example maximum thrust (positive for downward acceleration)
    
        # State and targets
        self.state = None
        self.target_xned = None
        self.target_euler = None


    def reset(self, 
              U: float = 30.0, 
              euler: np.ndarray = np.array([0, 0.0698, 0]), 
              xned: np.array = np.array([0, 0, 0]), 
              target_xned: Optional[np.array] = None,
              target_euler: Optional[np.array] = None):
        vned = np.array([U, 0, 0])
        vb = self.mavrik.ned2xyz(euler, vned)
        self.state = np.array([
            *vned,  # VXe, VYe, VZe
            *xned,   # Xe Ye Ze
            *vb,   # u v w
            *euler,   # roll, pitch, yaw 
            0.0, 0.0, 0.0,   # p, q, r
            0.0, 0.0, 0.0,   # Fx, Fy, Fz
            0.0, 0.0, 0.0    # L, M, N
        ])
        if target_xned is not None:
            self.target_xned = target_xned
        if target_euler is not None:
            self.target_euler = target_euler
            
        return self.state

    def normalize_thrust(self, *thrust_xyz):
        """
        Normalize raw thrust values to the range [-1, 1].
        """
        return np.clip((np.asarray(thrust_xyz) - self.thrust_xyz_min) / (self.thrust_xyz_max - self.thrust_xyz_min) * 2 - 1, -1, 1).tolist()
    
    def clip_control(self, control: np.ndarray) -> jnp.ndarray:
        """
        Clips the control inputs to ensure they are within valid ranges.

        Args:
            control (np.ndarray): Control array with 20 elements corresponding to self.mavrik.CONTROL.

        Returns:
            np.ndarray: Clipped control inputs.
        """
        # Wing and tail tilt angles [-pi/2, pi/2]
        control[self.mavrik.CONTROL.wing_tilt] = jnp.clip(control[self.mavrik.CONTROL.wing_tilt], -np.pi / 2, np.pi / 2)
        control[self.mavrik.CONTROL.tail_tilt] = jnp.clip(control[self.mavrik.CONTROL.tail_tilt], -np.pi / 2, np.pi / 2)
        
        # Aerodynamic control surfaces [-1, 1]
        control[self.mavrik.CONTROL.aileron] = jnp.clip(control[self.mavrik.CONTROL.aileron], -1, 1)
        control[self.mavrik.CONTROL.elevator] = jnp.clip(control[self.mavrik.CONTROL.elevator], -1, 1)
        control[self.mavrik.CONTROL.flap] = jnp.clip(control[self.mavrik.CONTROL.flap], -1, 1)
        control[self.mavrik.CONTROL.rudder] = jnp.clip(control[self.mavrik.CONTROL.rudder], -1, 1)
        
        # RPM values [0, 7500]
        for i in range(self.mavrik.CONTROL.RPM_tailLeft, self.mavrik.CONTROL.RPM_right12Out + 1):
            control[i] = jnp.clip(control[i], 0, 7500)
    
        return control
    
    def get_control(self):
        xned = self.state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]  # Extract xned ned (x, y, z)
        euler = self.state[self.mavrik.STATE.roll:self.mavrik.STATE.roll+3]  # Extract roll, pitch, yaw

        # Compute raw thrust values from PID controllers
        raw_thrust_x = self.pos_pid_x.compute(self.target_xned[0], xned[0])  # x_e (North)
        raw_thrust_y = self.pos_pid_y.compute(self.target_xned[1], xned[1])  # y_e (East)
        raw_thrust_z =  - self.pos_pid_z.compute(self.target_xned[2], xned[2])  # z_e (Down)

        # Normalize thrust values to [-1, 1]
        [thrust_x, thrust_y, thrust_z] = self.normalize_thrust(raw_thrust_x, raw_thrust_y, raw_thrust_z) 

        # Compute desired roll and pitch from thrusts
        roll_target = np.arctan(thrust_y / max(thrust_z, 1e-3))  # Prevent division by zero
        pitch_target = np.arctan(-thrust_x / max(thrust_z, 1e-3))

        # Compute euler control (desired moments or tilt angles)
        roll_control = self.att_pid_roll.compute(roll_target, euler[0])
        pitch_control = self.att_pid_pitch.compute(pitch_target, euler[1])
        yaw_control = self.att_pid_yaw.compute(self.target_euler[2], euler[2])


        # Map PID outputs to control inputs
        control = np.zeros(20)  # 20 control inputs
        control[self.mavrik.CONTROL.wing_tilt] = roll_control
        control[self.mavrik.CONTROL.tail_tilt] = pitch_control
        control[self.mavrik.CONTROL.rudder] = yaw_control

        # Compute base thrust RPMs
        rpm_base = thrust_z * 7500  # Normalize thrust to [0, 7500]
        pitch_adjustment = pitch_control * 100  # Example scaling for pitch
        roll_adjustment = roll_control * 100  # Example scaling for roll
        yaw_adjustment = yaw_control * 100  # Example scaling for yaw

        # Adjust RPMs for pitch (front and rear rotors)
        control[self.mavrik.CONTROL.RPM_leftOut1] = np.clip(rpm_base - pitch_adjustment, 0, 7500)  # Front left
        control[self.mavrik.CONTROL.RPM_left6In] = np.clip(rpm_base + pitch_adjustment, 0, 7500)  # Rear left
        control[self.mavrik.CONTROL.RPM_right7In] = np.clip(rpm_base + pitch_adjustment, 0, 7500)  # Rear right
        control[self.mavrik.CONTROL.RPM_right12Out] = np.clip(rpm_base - pitch_adjustment, 0, 7500)  # Front right

        # Adjust RPMs for roll (left and right rotors)
        control[self.mavrik.CONTROL.RPM_leftOut1] += np.clip(roll_adjustment, 0, 7500)
        control[self.mavrik.CONTROL.RPM_left6In] += np.clip(roll_adjustment, 0, 7500)
        control[self.mavrik.CONTROL.RPM_right7In] -= np.clip(roll_adjustment, 0, 7500)
        control[self.mavrik.CONTROL.RPM_right12Out] -= np.clip(roll_adjustment, -7500, 7500)

        # Adjust RPMs for yaw (tail rotors)
        control[self.mavrik.CONTROL.RPM_tailLeft] = np.clip(rpm_base + yaw_adjustment, 0, 7500)
        control[self.mavrik.CONTROL.RPM_tailRight] = np.clip(rpm_base - yaw_adjustment, 0, 7500)

        # Symmetric adjustment for mid rotors
        mid_rotors_left = range(self.mavrik.CONTROL.RPM_left2, self.mavrik.CONTROL.RPM_left6In + 1)
        mid_rotors_right = range(self.mavrik.CONTROL.RPM_right7In, self.mavrik.CONTROL.RPM_right11 + 1)
        for i in mid_rotors_left:
            control[i] = np.clip(rpm_base + pitch_adjustment - roll_adjustment, 0, 7500)
        for i in mid_rotors_right:
            control[i] = np.clip(rpm_base + pitch_adjustment + roll_adjustment, 0, 7500)
                    
        # Clip control inputs
        control = self.clip_control(control)
        return control



def run_pid_and_plot_trajectories(pid_controller, initial_conditions, target_conditions, max_steps=100, run_name=''):
    assert len(initial_conditions) == len(target_conditions), \
        f"Number of initial and target conditions do not match: {len(initial_conditions)=} vs. {len(target_conditions)=}"

    run_name = f"{run_name}_pid_{max_steps}steps_{len(initial_conditions)}trajectories_{get_timestamp()}"
    print(f"Running {run_name}...")
    trajectories = []
    for initial_condition, target_condition in zip(initial_conditions, target_conditions):
        U, euler, xned = initial_condition['U'], initial_condition['euler'], initial_condition['xned']
        target_xned, target_euler = target_condition['xned'], target_condition['euler']
        state = pid_controller.reset(U=U, euler=euler, xned=xned, target_xned=target_xned, target_euler=target_euler)
        trajectory = {'state': [], 'control': []}

        for _ in range(max_steps):
            if jnp.isnan(state).any() or \
                state[pid_controller.mavrik.STATE.Ze] > 10 or \
                    state[pid_controller.mavrik.STATE.Ze] < -200 or \
                        (state[pid_controller.mavrik.STATE.u: pid_controller.mavrik.STATE.u+3] > 100).any():
                break
            
            control = pid_controller.get_control()
            if jnp.isnan(control).any():
                break
            
            trajectory['state'].append(state)  # (altitude, pitch angle)
            trajectory['control'].append(control)
            
            state, _ = pid_controller.mavrik.step(state, control)
                                   
        trajectories.append(trajectory)

    
    

    # Save trajectories to a pickle file
    trajectory_dir = os.path.join(current_dir, 'data')
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    with open(f'{trajectory_dir}/{run_name}.pt', 'wb') as f:
        pickle.dump(trajectories, f)
    

    # Plotting
    plt.figure()
     
    plt.figure()
    for trajectory in trajectories:
        states = np.array(trajectory['state'])
        altitudes = -states[:, pid_controller.mavrik.STATE.Ze]
        pitches = states[:, pid_controller.mavrik.STATE.pitch]
        velocities = states[:, pid_controller.mavrik.STATE.VXe: pid_controller.mavrik.STATE.VXe+3]
        speeds = np.linalg.norm(velocities, axis=-1)
        

        plt.subplot(3, 1, 1)
        plt.plot(altitudes, pitches, color='b')
        plt.scatter(altitudes[0], pitches[0], color='green', s=50)
        plt.scatter(altitudes[-1], pitches[-1], color='black', s=50)
        plt.xlabel('Altitude')
        plt.ylabel('Pitch Angle')

        plt.subplot(3, 1, 2)
        plt.plot(altitudes, speeds, color='r')
        plt.scatter(altitudes[0], speeds[0], color='green', s=50)
        plt.scatter(altitudes[-1], speeds[-1], color='black', s=50)
        plt.xlabel('Altitude')
        plt.ylabel('Speed')

        plt.subplot(3, 1, 3)
        plt.plot(altitudes, velocities[:, -1], color='r')
        plt.scatter(altitudes[0], velocities[0][-1], color='green', s=50)
        plt.scatter(altitudes[-1], velocities[-1][-1], color='black', s=50)
        plt.xlabel('Altitude')
        plt.ylabel('Vertical Velocity')

    plt.tight_layout()
    
    plot_dir = os.path.join(current_dir, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(f'{plot_dir}/{run_name}.png')
 
    return run_name
    
# Example usage
if __name__ == "__main__":

    print(f"JAX is using: {xla_bridge.get_backend().platform}")

    # Ensure PID runs on CPU
    pid_controller = MavrikPIDController(Mavrik(), pos_K = np.random.random([3, 3]) * 2, att_K = np.random.random([3, 3])  * 2, dt=0.01)
    data_path = os.path.join(current_dir, 'data')

    for max_steps in [10000]: #10, 100, 500, 1000]:
        for num_trajs in [100]: #, 100, 500, 1000]:
            initial_conditions = [
                #{'U': 30, 'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, np.random.uniform(-100, -50)])}
                {'U': 0, 'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, 0])}
                for i in range(num_trajs)
            ]
            target_conditions = [
                #{'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, 0])}
                {'U': 0, 'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, np.random.uniform(-100, -50)])}
                for i in range(num_trajs)
            ]

            run_pid_and_plot_trajectories(pid_controller, initial_conditions, target_conditions, max_steps=max_steps, run_name = 'hover')

    # Ensure SystemID runs on GPU
    with jax.default_device(jax.devices("gpu")[0]):
        system_id = SystemID()
        system_id.run('hover_1000x100_system_id')
