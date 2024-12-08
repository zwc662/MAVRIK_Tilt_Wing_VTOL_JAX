import gym
import numpy as np 

from typing import Optional

from jax_mavrik.mavrik import Mavrik

import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level to info
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d')

# Add formatter to console handler
ch.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(ch)


class MavrikEnv(gym.Env):
    def __init__(self):
        super(MavrikEnv, self).__init__()
        self.mavrik = Mavrik()
        self.target_altitude = 2.0
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.mavrik.state_ndim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.max_speed = 5.
        self.min_altitude = -1
        self.max_altitude = 5

        self.state = self.reset()
    
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
        for i in range(self.mavrik.CONTROL.RPM_left2, self.mavrik.CONTROL.RPM_left6In + 1):
            control[i] = np.clip(rpm_base + pitch_adjustment - roll_adjustment, 0, 7500)
        for i in range(self.mavrik.CONTROL.RPM_right7In, self.mavrik.CONTROL.RPM_right11 + 1):
            control[i] = np.clip(rpm_base + pitch_adjustment + roll_adjustment, 0, 7500)
                    
        # Clip control inputs
        control = self.clip_control(control)
        return np.array(control)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        control = self.get_control(*action)
         
        next_state, info = self.mavrik.step(self.state, control)
        
        done = self._is_done(next_state)

        if done < 0:
            reward = -1
        else:    
            reward = self._compute_climbing_reward_from_next_state(next_state)    

        self.state = next_state
 
        return next_state, reward, done, info

    def reset(self, 
              vned: Optional[np.ndarray] = None, 
              euler: Optional[np.ndarray] = None,
              xned:  Optional[np.ndarray] = None,
              target_altitude: Optional[np.array] = None
              ):
        if vned is None:
            vned = np.array([0., 0., 0.])
        if euler is None:
            euler = np.array([0, 0.0698, 0]) + (np.random.random([3,]) - 0.5) * np.pi/6.0 
        if xned is None:
            xned = np.array([0., 0., 0.]) 
                            
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
        else:
            self.target_altitude = 2.0
        
        return self.state

    def render(self, mode='human'):
        self.mavrik.render(mode)

    def _compute_reward_from_state_control_next_state(self, state, control, next_state):
        altitude = - state[self.mavrik.STATE.Ze]  # Assuming the first three elements of the state are the position
        next_altitude = - next_state[self.mavrik.STATE.Ze]
        distance = np.abs(altitude - self.target_altitude)
        next_distance = np.abs(next_altitude - self.target_altitude)

        if next_distance > distance:
            reward = 0 
        else:
            reward = np.exp(1e3 * (distance - next_distance)) #square_velocity / (3 * self.max_speed**2) # The closer to the target, the greater the reward (less negative)
        
        return reward
    
    def _compute_hover_reward_from_next_state(self, next_state):
        verticle_speed = np.abs(next_state[self.mavrik.STATE.VZe]).item()
        reward = - verticle_speed / self.max_speed #square_velocity / (3 * self.max_velocity**2) # The closer to the target, the greater the reward (less negative)
        return reward
    
    def _compute_climbing_reward_from_next_state(self, next_state):
        altitude = - next_state[self.mavrik.STATE.Ze]  # Assuming the first three elements of the state are the position
        distance = altitude - self.target_altitude
        reward =  np.exp(- np.abs(distance)) #square_velocity / (3 * self.max_speed**2) # The closer to the target, the greater the reward (less negative)
        return reward

    def _is_done(self, state):
        if np.isnan(state).any():
            logger.warning(f"NaN encountered in state: {state}. Ending episode.")
            return -1
        if False and self.mavrik.error_check(state):
            logger.warning(f"Error in next state: {next_state}. Ending episode.")
            return -1
            
        position = state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]
        altitude = - state[self.mavrik.STATE.Ze]
        distance = np.abs(altitude - self.target_altitude)
        velocity = state[self.mavrik.STATE.VXe:self.mavrik.STATE.VXe+3]
        max_speed =  np.abs(velocity).max()

        if altitude > self.max_altitude:
            logger.warning(f"Altitude {altitude} >= max altitude {self.max_altitude}. Ending episode.")
            return -1
        if altitude < self.min_altitude:
            logger.warning(f"Altitude {altitude} <= min altitude {self.min_altitude}. Ending episode.")
            return -1
        if max_speed > self.max_speed:
            logger.warning(f"Velocity {velocity} > max speed {self.max_speed}. Ending episode.")
            return -1
        
        return 0
        #if  distance < 1.0 and speed < 5.0:
        #    #print("Done Done Done")
        #    return 0 # Consider done if within 5 unit of the target while speed below 5 units
        #else:
        #    return 0
         

    
 