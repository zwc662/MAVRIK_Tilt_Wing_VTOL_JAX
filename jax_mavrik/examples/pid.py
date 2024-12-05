import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

from jax_mavrik.mavrik import Mavrik
 
import numpy as np
import datetime

import os
import sys
import pickle

from jax_mavrik.examples.system_id import SystemID
 
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
    def __init__(self, mavrik: Mavrik, dt: float = 0.01):
        self.mavrik = mavrik
        self.dt = dt

        self.pos_pid_x = None
        self.pos_pid_y = None
        self.pos_pid_z = None
        self.att_pid_roll = None
        self.att_pid_pitch = None
        self.att_pid_yaw = None
        self.thrust_xyz_min = None
        self.thrust_xyz_max = None
        self.target_xned = None
        self.target_euler = None
         
        
    def create_pid(self, pos_K, att_K):
        # PID controllers for ned xned (x, y, z)
        self.pos_pid_x = PIDController(self.dt, *pos_K[0])
        self.pos_pid_y = PIDController(self.dt, *pos_K[1])
        self.pos_pid_z = PIDController(self.dt, *pos_K[2])

        # PID controllers for orientation (roll, pitch, yaw)
        self.att_pid_roll = PIDController(self.dt, *att_K[0])
        self.att_pid_pitch = PIDController(self.dt, *att_K[1])
        self.att_pid_yaw = PIDController(self.dt, *att_K[2])

        # Define expected thrust ranges for normalization
        self.thrust_xyz_min = np.array([-10.0, -10.0, 0.0])  # Example minimum thrust (negative for upward acceleration)
        self.thrust_xyz_max = np.array([10.0, 10.0, 10.0])   # Example maximum thrust (positive for downward acceleration)
     
    def reset(self, 
            U: float = 30.0, 
            euler: np.ndarray = np.array([0, 0.0698, 0]), 
            xned: np.array = np.array([0, 0, 0]), 
            target_xned: Optional[np.array] = None,
            target_euler: Optional[np.array] = None,
            pos_K: Optional[np.array] = None,
            att_K: Optional[np.array] = None
    ):
        if pos_K is None:
            pos_K = np.array([[1.0, 0.1, 0.5], [1.0, 0.1, 0.5], [1.5, 0.2, 0.6]]) + np.random.random([3, 3]) * 2
        
        if att_K is None:
            att_K = np.array([[1.2, 0.1, 0.4], [1.2, 0.1, 0.4], [1.5, 0.2, 0.5]]) + np.random.random([3, 3]) * 2
        
        self.create_pid(
            pos_K = pos_K,
            att_K = att_K,
        )

        vned = np.array([U, 0, 0])
        vb = self.mavrik.ned2xyz(euler, vned)
        state = np.array([
            *vned,  # VXe, VYe, VZe
            *xned,   # Xe Ye Ze
            *vb,   # u v w
            *euler,   # roll, pitch, yaw 
            0.0, 0.0, 0.0,   # p, q, r
            0.0, 0.0, 0.0,   # Fx, Fy, Fz
            0.0, 0.0, 0.0    # L, M, N
        ])

        self.target_xned = target_xned
        if target_xned is not None:
            self.target_xned = np.array([0, 0, np.random.uniform(-10, -5)])
        
        self.target_euler = target_euler
        if target_euler is not None:
            self.target_euler = np.array([0, 0, 0])

        return state

    def normalize_thrust(self, *thrust_xyz):
        """
        Normalize raw thrust values to the range [-1, 1].
        """
        return np.clip((np.asarray(thrust_xyz) - self.thrust_xyz_min) / (self.thrust_xyz_max - self.thrust_xyz_min) * 2 - 1, -1, 1).tolist()
    
    def clip_control(self, control: np.ndarray) -> np.ndarray:
        """
        Clips the control inputs to ensure they are within valid ranges.

        Args:
            control (np.ndarray): Control array with 20 elements corresponding to self.mavrik.CONTROL.

        Returns:
            np.ndarray: Clipped control inputs.
        """
        # Wing and tail tilt angles [-pi/2, pi/2]
        control[self.mavrik.CONTROL.wing_tilt] = np.clip(control[self.mavrik.CONTROL.wing_tilt], -np.pi / 2, np.pi / 2)
        control[self.mavrik.CONTROL.tail_tilt] = np.clip(control[self.mavrik.CONTROL.tail_tilt], -np.pi / 2, np.pi / 2)
        
        # Aerodynamic control surfaces [-1, 1]
        control[self.mavrik.CONTROL.aileron] = np.clip(control[self.mavrik.CONTROL.aileron], -1, 1)
        control[self.mavrik.CONTROL.elevator] = np.clip(control[self.mavrik.CONTROL.elevator], -1, 1)
        control[self.mavrik.CONTROL.flap] = np.clip(control[self.mavrik.CONTROL.flap], -1, 1)
        control[self.mavrik.CONTROL.rudder] = np.clip(control[self.mavrik.CONTROL.rudder], -1, 1)
        
        # RPM values [0, 7500]
        for i in range(self.mavrik.CONTROL.RPM_tailLeft, self.mavrik.CONTROL.RPM_right12Out + 1):
            control[i] = np.clip(control[i], 0, 7500)
    
        return control
    
    def get_control(self, state: np.ndarray) -> np.ndarray:
        xned = state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]  # Extract xned ned (x, y, z)
        euler = state[self.mavrik.STATE.roll:self.mavrik.STATE.roll+3]  # Extract roll, pitch, yaw

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

        # Adjust RPMs for pitch (front and rear rotors), roll (left and right rotors)
        control[self.mavrik.CONTROL.RPM_leftOut1] = np.clip(rpm_base - pitch_adjustment + roll_adjustment, 0, 7500)  # Front left
        control[self.mavrik.CONTROL.RPM_left6In] = np.clip(rpm_base + pitch_adjustment + roll_adjustment, 0, 7500)  # Rear left
        control[self.mavrik.CONTROL.RPM_right7In] = np.clip(rpm_base + pitch_adjustment - roll_adjustment, 0, 7500)  # Rear right
        control[self.mavrik.CONTROL.RPM_right12Out] = np.clip(rpm_base - pitch_adjustment - roll_adjustment, 0, 7500)  # Front right
 
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



def run_pid_and_plot_trajectories(pid_controller, initial_conditions, target_conditions, pos_K, att_K, max_steps=100, run_name=''):
    assert len(initial_conditions) == len(target_conditions), \
        f"Number of initial and target conditions do not match: {len(initial_conditions)=} vs. {len(target_conditions)=}"

    run_name = f"{run_name}_pid_{max_steps}steps_{len(initial_conditions)}trajectories_{get_timestamp()}"
    print(f"Running {run_name}...")
    trajectories = []
    for initial_condition, target_condition in zip(initial_conditions, target_conditions):
        if len(trajectories) % (len(initial_conditions) // 10) == 0:
            print(f"Collected {len(trajectories)} trajectories")
        U, euler, xned = initial_condition['U'], initial_condition['euler'], initial_condition['xned']
        target_xned, target_euler = target_condition['xned'], target_condition['euler']
        state = pid_controller.reset(U=U, euler=euler, xned=xned, target_xned=target_xned, target_euler=target_euler, pos_K = pos_K, att_K = att_K)
        trajectory = {'state': [], 'control': []}

         
        for step in range(max_steps):
            if np.isnan(state).any() or \
                state[pid_controller.mavrik.STATE.Ze] > 10 or \
                    np.abs(state[pid_controller.mavrik.STATE.Ze]) > 1e6 or \
                        (np.abs(state[pid_controller.mavrik.STATE.u: pid_controller.mavrik.STATE.u+3]) > 1e6).any() or \
                            pid_controller.mavrik.error_check(state):
                #input()
                break
            
            control = pid_controller.get_control(state)
            if False:
                control = np.array([
                    0.0, 0.0, 0.0,  # wing_tilt, tail_tilt, aileron
                    0.0, 0.0, 0.0,  # elevator, flap, rudder
                    7500.0, 7500.0,  # RPM_tailLeft, RPM_tailRight
                    7500.0, 7500.0,  # RPM_leftOut1, RPM_left2
                    7500.0, 7500.0,  # RPM_left3, RPM_left4
                    7500.0, 7500.0,  # RPM_left5, RPM_left6In
                    7500.0, 7500.0,  # RPM_right7In, RPM_right8
                    7500.0, 7500.0,  # RPM_right9, RPM_right10
                    7500.0, 7500.0   # RPM_right11, RPM_right12Out
                ])
                
            if np.isnan(control).any():
                break
            
            trajectory['state'].append(state)  # (altitude, pitch angle)
            trajectory['control'].append(control)

            if False: #True:
                print(f">>>>>>>>>>>>>>>>>>>> Step: {step} <<<<<<<<<<<<<<<<<<<<<<") 
                vned = state[pid_controller.mavrik.STATE.VXe:pid_controller.mavrik.STATE.VXe+3]
                print(f"Vned: {vned}")# | Expected Vned: {expected_vned[i]} | Error: {np.linalg.norm(vned - expected_vned[i])}")
                xned = state[pid_controller.mavrik.STATE.Xe:pid_controller.mavrik.STATE.Xe+3]
                print(f"Xned: {xned}")# | Expected Xned: {expected_xned[i]} | Error: {np.linalg.norm(xned - expected_xned[i])}")
                vb = state[pid_controller.mavrik.STATE.u:pid_controller.mavrik.STATE.u+3]
                print(f"Vb: {vb}")# | Expected Vb: {expected_vb[i]} | Error: {np.linalg.norm(vb - expected_vb[i])}")
                euler = state[pid_controller.mavrik.STATE.roll:pid_controller.mavrik.STATE.roll+3]
                print(f"Euler: {euler}")# | Expected Euler: {expected_euler[i]} | Error: {np.linalg.norm(euler - expected_euler[i])}")
                pqr = state[pid_controller.mavrik.STATE.p:pid_controller.mavrik.STATE.p+3]
                print(f"PQR: {pqr}")
                forces = state[pid_controller.mavrik.STATE.Fx:pid_controller.mavrik.STATE.Fx+3]
                print(f"Forces: {forces}")
                moments = state[pid_controller.mavrik.STATE.L:pid_controller.mavrik.STATE.L+3]
                print(f"Moments: {moments}")
            
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
        plt.plot(altitudes, -velocities[:, -1], color='r')
        plt.scatter(altitudes[0], -velocities[0][-1], color='green', s=50)
        plt.scatter(altitudes[-1], -velocities[-1][-1], color='black', s=50)
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
    # Ensure PID runs on CPU
    pid_controller = MavrikPIDController(Mavrik(), dt=0.01)
    data_path = os.path.join(current_dir, 'data')

    for max_steps in [10000]: #10, 100, 500, 1000]:
        for num_trajs in [10]: #, 100, 500, 1000]:
            initial_conditions = [
                #{'U': 30, 'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, np.random.uniform(-100, -50)])}
                {'U': 0, 'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, 0])}
                for i in range(num_trajs)
            ]
            target_conditions = [
                #{'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, 0])}
                {'U': 0, 'euler': np.array([0, 0.0698, 0]), 'xned': np.array([0, 0, -20])} #np.random.uniform(-10, -5)])}
                for i in range(num_trajs)
            ]
            pos_K = np.array([[1.0, 0.1, 0.5], [1.0, 0.1, 0.5], [1.5, 0.2, 0.6]]) # + np.random.random([3, 3]) * 2
            att_K = np.array([[1.2, 0.1, 0.4], [1.2, 0.1, 0.4], [1.5, 0.2, 0.5]]) #+ np.random.random([3, 3]) * 2


            run_pid_and_plot_trajectories(pid_controller, initial_conditions, target_conditions, pos_K, att_K, max_steps=max_steps, run_name = 'hover')

    # Ensure SystemID runs on GPU
    system_id = SystemID()
    system_id.run('hover_1000x100_system_id')
