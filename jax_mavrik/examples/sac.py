import gym
import numpy as np
import jax.numpy as jnp

from typing import Optional
from jax_mavrik.mavrik import Mavrik
import jax
import optax
from flax import linen as nn
from flax.training import train_state

class MavrikEnv(gym.Env):
    def __init__(self):
        super(MavrikEnv, self).__init__()
        self.mavrik = Mavrik()
        self.target_position = np.array([100., 100., 0.])
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.mavrik.state_ndim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.mavrik.control_ndim,), dtype=np.float32)
        self.state = self.reset()

    def step(self, action):
        control = np.zeros(self.mavrik.control_ndim) + action
        control= control.at[7:].set((control[7:] + 1) * 3750)
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
              pqr: np.array = np.array([0, 0.0698, 0]),
              target_xned: Optional[np.array] = None):
        vned = np.array([U, 0, 0])
        vb = self.mavrik.ned2xyz(euler, vned)
        self.state = np.array([
            *vned,  # VXe, VYe, VZe
            *xned,   # Xe Ye Ze
            *vb,   # u v w
            *euler,   # roll, pitch, yaw 
            0, 0.0698, 0,   # wx, wy, wz 
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
            print("Encountered NaN in state, ending episode.")
            return -1
            
        position = state[self.mavrik.STATE.Xe:self.mavrik.STATE.Xe+3]
        distance = np.linalg.norm(position - self.target_position)
        velocity = state[self.mavrik.STATE.VXe:self.mavrik.STATE.VXe+3]
        speed =  np.linalg.norm(velocity)
        return int(distance < 5.0 and speed < 5.0) # Consider done if within 5 unit of the target while speed below 5 units
         

class Actor(nn.Module):
    action_dim: int

    def setup(self):
        self.dense1 = nn.Dense(256)
        self.dense2 = nn.Dense(256)
        self.dense3 = nn.Dense(self.action_dim)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        return nn.tanh(self.dense3(x))

class Critic(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(256)
        self.dense2 = nn.Dense(256)
        self.dense3 = nn.Dense(1)

    def __call__(self, x, a):
        x = jnp.concatenate([x, a], axis=-1)
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        return self.dense3(x)

class SAC:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.actor = Actor(action_dim)
        self.critic1 = Critic()
        self.critic2 = Critic()
        self.target_critic1 = Critic()
        self.target_critic2 = Critic()

        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(jax.random.PRNGKey(0), jnp.ones((1, state_dim))),
            tx=optax.adam(lr)
        )
        self.critic1_state = train_state.TrainState.create(
            apply_fn=self.critic1.apply,
            params=self.critic1.init(jax.random.PRNGKey(1), jnp.ones((1, state_dim)), jnp.ones((1, action_dim))),
            tx=optax.adam(lr)
        )
        self.critic2_state = train_state.TrainState.create(
            apply_fn=self.critic2.apply,
            params=self.critic2.init(jax.random.PRNGKey(2), jnp.ones((1, state_dim)), jnp.ones((1, action_dim))),
            tx=optax.adam(lr)
        )
        self.target_critic1_state = self.critic1_state
        self.target_critic2_state = self.critic2_state

    def select_action(self, state):
        action = self.actor.apply(self.actor_state.params, state)
        return action

    def update(self, state, action, reward, next_state, done):
        def critic_loss_fn(params, state, action, reward, next_state, done):
            q1 = self.critic1.apply(params, state, action)
            q2 = self.critic2.apply(params, state, action)
            next_action = self.actor.apply(self.actor_state.params, next_state)
            next_q1 = self.target_critic1.apply(self.target_critic1_state.params, next_state, next_action)
            next_q2 = self.target_critic2.apply(self.target_critic2_state.params, next_state, next_action)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = reward + (1.0 - done) * 0.99 * next_q
            loss1 = jnp.mean((q1 - target_q) ** 2)
            loss2 = jnp.mean((q2 - target_q) ** 2)
            return loss1 + loss2, (q1, q2, next_q, target_q)
        
        def actor_loss_fn(params, state):
            action = self.actor.apply(params, state)
            q1 = self.critic1.apply(self.critic1_state.params, state, action)
            q2 = self.critic2.apply(self.critic2_state.params, state, action)
            q = jnp.minimum(q1, q2)
            return -jnp.mean(q), (action, q1, q2)
        
        (critic_loss, critic_aux), critic_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(self.critic1_state.params, state, action, reward, next_state, done)
        self.critic1_state = self.critic1_state.apply_gradients(grads=critic_grads)
        self.critic2_state = self.critic2_state.apply_gradients(grads=critic_grads)

        (actor_loss, actor_aux), actor_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(self.actor_state.params, state)
        self.actor_state = self.actor_state.apply_gradients(grads=actor_grads)

        self.target_critic1_state = self.target_critic1_state.replace(
        params=optax.incremental_update(self.critic1_state.params, self.target_critic1_state.params, 0.005)
        )
        self.target_critic2_state = self.target_critic2_state.replace(
        params=optax.incremental_update(self.critic2_state.params, self.target_critic2_state.params, 0.005)
        )
         
                   
if __name__ == "__main__":
    env = MavrikEnv()
    sac = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    num_episodes=1_000_000_000
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = sac.select_action(state)
            next_state, reward, done, _ = env.step(action)
            sac.update(state, action, reward, next_state, done)
            state = next_state

            
            print(f"Episode: {episode}, Reward: {reward}, Done: {done}")
            

                
 