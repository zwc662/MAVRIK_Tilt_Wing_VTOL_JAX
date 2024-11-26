import gym
import numpy as np
import jax.numpy as jnp

from typing import Optional

from jax_mavrik.mavrik import Mavrik
 

class MavrikEnv(gym.Env):
    def __init__(self):
        super(MavrikEnv, self).__init__()
        self.mavrik = Mavrik()
        self.target_altitude = 0.0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.mavrik.state_ndim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.max_velocity = 50.
        self.state = self.reset()
    
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
    
    def get_control(self, 
                    roll_control: float, 
                    pitch_control: float,
                    yaw_control: float,
                    thrust_z: float
                    ) -> np.ndarray:
        # Map PID outputs to control inputs
        control = np.zeros(self.mavrik.control_ndim)  # 20 control inputs
        control[self.mavrik.CONTROL.wing_tilt] = roll_control
        control[self.mavrik.CONTROL.tail_tilt] = pitch_control
        control[self.mavrik.CONTROL.rudder] = yaw_control

        # Compute base thrust RPMs
        rpm_base = (thrust_z + 1.) * 7500 / 2.  # Normalize thrust to [0, 7500]
        pitch_adjustment = pitch_control * 100  # Example scaling for pitch
        roll_adjustment = roll_control * 100  # Example scaling for roll
        yaw_adjustment = yaw_control * 100  # Example scaling for yaw

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
        return jnp.array(control)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        control = self.get_control(*action)
         
        next_state, info = self.mavrik.step(self.state, control)
        reward = self._compute_reward(next_state)
        self.state = next_state

        done = self._is_done(next_state, control)
        if done < 0:
            reward = -1
            self.state = self.reset()
            
        return next_state, reward, done, info

    def reset(self, 
              U: float = 0.0, 
              euler: np.ndarray = np.array([0, 0.0698, 0]), 
              xned: np.array = np.array([0, 0, 0]), 
              target_altitude: Optional[np.array] = np.random.random() * 50 + 50 #np.array([100, 0, 0])
              ): #np.array([0, 0, 0])):
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
        if target_altitude is not None:
            self.target_altitude = target_altitude
        
        return self.state

    def render(self, mode='human'):
        self.mavrik.render(mode)

    def _compute_reward(self, state):
        altitude = state[self.mavrik.STATE.Ze]  # Assuming the first three elements of the state are the position
        distance = np.maximum(0, altitude - self.target_altitude)
        square_velocity = np.sum(state[self.mavrik.STATE.VXe:self.mavrik.STATE.VXe+3]**2)
        reward = np.exp(-distance) - square_velocity / (3 * self.max_velocity**2) # The closer to the target, the greater the reward (less negative)
        
        return reward

    def _is_done(self, state, control):
        next_state, _ = self.mavrik.step(self.state, control)
        if jnp.isnan(state).any():
            print(f"NaN encountered in state: {next_state}. Ending episode.")
            return -1
            
        position = state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]
        altitude = state[self.mavrik.STATE.Ze]
        distance = np.abs(altitude - self.target_altitude)
        velocity = state[self.mavrik.STATE.VXe:self.mavrik.STATE.VXe+3]
        speed =  np.linalg.norm(velocity)

        if state[self.mavrik.STATE.Ze] > 10 or \
            state[self.mavrik.STATE.Ze] < -200 or \
                (state[self.mavrik.STATE.u: self.mavrik.STATE.u+3] > self.max_velocity).any():
            print("Out of bounds. Ending episode.")
            return -1
        if  distance < 5.0 and speed < 5.0:
            #print("Done Done Done")
            return 0 # Consider done if within 5 unit of the target while speed below 5 units
        else:
            return 0
         