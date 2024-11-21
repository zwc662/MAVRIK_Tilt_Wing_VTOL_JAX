import gym
import numpy as np
import jax.numpy as jnp

from typing import Optional

from jax_mavrik.mavrik import Mavrik
 

class MavrikEnv(gym.Env):
    def __init__(self):
        super(MavrikEnv, self).__init__()
        self.mavrik = Mavrik()
        self.target_position = np.array([100., 100., 0.])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.mavrik.state_ndim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.mavrik.control_ndim,), dtype=np.float32)
        self.state = self.reset()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        control = jnp.zeros(self.mavrik.control_ndim) + action
        control= control.at[self.mavrik.CONTROL.RPM_tailLeft:].set((control[self.mavrik.CONTROL.RPM_tailLeft:] + 1) * 3750)
        control = control.at[self.mavrik.CONTROL.flap].set((control[self.mavrik.CONTROL.flap] + 1)/ 2.)
        control = control.at[self.mavrik.CONTROL.wing_tilt].set(control[self.mavrik.CONTROL.wing_tilt] * jnp.pi / 4.)
        control = control.at[self.mavrik.CONTROL.tail_tilt].set(control[self.mavrik.CONTROL.tail_tilt] * jnp.pi / 4.)
 
        next_state, info = self.mavrik.step(self.state, control)
        reward = self._compute_reward(next_state)
        self.state = next_state

        done = self._is_done(next_state, control)
        if done < 0:
            reward = -1
            self.state = self.reset()
            
        return next_state, reward, done, info

    def reset(self, 
              U: float = 30.0, 
              euler: np.ndarray = np.array([0, 0.0698, 0]), 
              xned: np.array = np.array([0, 0, 0]), 
              target_xned: Optional[np.array] = None):
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
            self.target_position = target_xned
            
        return self.state

    def render(self, mode='human'):
        self.mavrik.render(mode)

    def _compute_reward(self, state):
        position = state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]  # Assuming the first three elements of the state are the position
        distance = np.linalg.norm(position - self.target_position)
        reward = np.exp(-distance)  # The closer to the target, the greater the reward (less negative)
        return reward

    def _is_done(self, state, control):
        next_state, _ = self.mavrik.step(self.state, control)
        if np.isnan(next_state).any():
            print(f"NaN encountered in state: {next_state}. Ending episode.")
            return -1
            
        position = state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]
        distance = np.linalg.norm(position - self.target_position)
        velocity = state[self.mavrik.STATE.VXe:self.mavrik.STATE.VXe+3]
        speed =  np.linalg.norm(velocity)
        return int(distance < 5.0 and speed < 5.0) # Consider done if within 5 unit of the target while speed below 5 units
         