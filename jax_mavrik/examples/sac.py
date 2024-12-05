import gym
import numpy as np
import jax.numpy as jnp

from typing import Optional 
from jax_mavrik.examples.gym_mavrik import MavrikEnv
import jax
import optax
from flax import linen as nn
from flax.training import train_state
import os
import pickle

import datetime
import logging

import matplotlib.pyplot as plt

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Default level

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - Line: %(lineno)d')

# Add formatter to ch
ch.setFormatter(formatter)

# Add ch to logger
logger.addHandler(ch)

def set_logging_level(level: str):
    level = level.upper()
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)
    elif level == 'CRITICAL':
        logger.setLevel(logging.CRITICAL)
    else:
        raise ValueError(f"Unknown logging level: {level}")

current_dir = os.path.dirname(os.path.abspath(__file__))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



class Actor(nn.Module):
    action_dim: int

    def setup(self):
        self.dense1 = nn.Dense(256)
        self.dense2 = nn.Dense(256)
        self.mean_layer = nn.Dense(self.action_dim)
        self.log_std_layer = nn.Dense(self.action_dim)

    def __call__(self, x):
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        mean = jnp.tanh(self.mean_layer(x))
        log_std = self.log_std_layer(x)
        log_std = jnp.clip(log_std, -20, 2)  # Clipping log_std to prevent numerical issues
        std = jnp.exp(log_std)
        return mean, std

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
        mean, std = self.actor.apply(self.actor_state.params, state)
        action = mean + std * jax.random.normal(jax.random.PRNGKey(0), mean.shape)
        action = jnp.tanh(action)  # Squash the action into [-1, 1]
        return mean #action

    def update(self, state, action, reward, next_state, done):
        def critic_loss_fn(params, state, action, reward, next_state, done):
            q1 = self.critic1.apply(params, state, action)
            q2 = self.critic2.apply(params, state, action)
            next_action = self.select_action(next_state)
            next_q1 = self.target_critic1.apply(self.target_critic1_state.params, next_state, next_action)
            next_q2 = self.target_critic2.apply(self.target_critic2_state.params, next_state, next_action)
            next_q = jnp.minimum(next_q1, next_q2)
            target_q = reward + (1.0 - done) * 0.99 * next_q
            loss1 = jnp.mean((q1 - target_q) ** 2)
            loss2 = jnp.mean((q2 - target_q) ** 2)
            return loss1 + loss2, (q1, q2, next_q, target_q)
        
        def actor_loss_fn(params, state):
            mean, std = self.actor.apply(params, state)
            action = mean + std * jax.random.normal(jax.random.PRNGKey(0), mean.shape)
            action = jnp.tanh(action)  # Squash the action into [-1, 1]
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
         
                   
def run(max_steps = 1000, num_episodes = 1_000_000_000):
    env = MavrikEnv()
    sac = SAC(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    
    for episode in range(num_episodes):
        state = env.reset()
        done = 1
        episode_reward = 0
        steps = 0
        while steps <= max_steps:
            action = sac.select_action(state)
            next_state, reward, done, _ = env.step(action)
            if done < 0:
                sac.update(state, action, reward, state, done)
                break
            else:
                sac.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            steps += 1
        logger.info(f"Episode: {episode}, Reward: {episode_reward}, Done: {done}")

        if episode % 20 == 0:
            trajectories = []

            for _ in range(10):
                state = env.reset()
                done = 1
                trajectory = {'state': [], 'action': [], 'reward': []}
                total_reward = 0
                steps = 0
                while steps <= max_steps:
                    action = sac.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    trajectory['state'].append(state)
                    trajectory['action'].append(action)
                    trajectory['reward'].append(reward)
                    total_reward += reward
                    if done < 0:
                        break
                    else:
                        state = next_state
                    steps += 1
                trajectories.append(trajectory)
            
            # Compute average total reward
            average_total_reward = np.mean([sum(t['reward']) for t in trajectories])
            logger.info(f"Average Total Reward: {average_total_reward}")

            # Save trajectories to a pickle file
            trajectory_dir = os.path.join(current_dir, 'data')
            if not os.path.exists(trajectory_dir):
                os.makedirs(trajectory_dir)
            with open(f'{trajectory_dir}/hover_sac_10trajectories_{timestamp}.pt', 'wb') as f:
                pickle.dump(trajectories, f)
 
            # Plotting
            plt.figure(figsize=(12, 8))
            for trajectory in trajectories:
                states = np.array(trajectory['state'])
                altitudes = - states[:, env.mavrik.STATE.Ze]
                pitches = states[:, env.mavrik.STATE.pitch]
                speeds = np.sqrt(np.sum(states[:, env.mavrik.STATE.VXe:env.mavrik.STATE.VZe+1] ** 2, axis=1))
                vertical_velocities = -states[:, env.mavrik.STATE.VZe]

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
                plt.plot(altitudes, vertical_velocities, color='r')
                plt.scatter(altitudes[0], vertical_velocities[0], color='green', s=50)
                plt.scatter(altitudes[-1], vertical_velocities[-1], color='black', s=50)
                plt.xlabel('Altitude')
                plt.ylabel('Vertical Velocity')

            plt.tight_layout()
            plot_dir = os.path.join(current_dir, 'plots')
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(f'{plot_dir}/hover_sac_10trajectories_ep{episode}_{timestamp}.png')
            #plt.show()
            
            # Save the model parameters
            model_dir = os.path.join(current_dir, 'models')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            with open(f'{model_dir}/hover_sac_actor_params_{timestamp}.pkl', 'wb') as f:
                pickle.dump(sac.actor_state.params, f)
            with open(f'{model_dir}/hover_sac_critic1_params_{timestamp}.pkl', 'wb') as f:
                pickle.dump(sac.critic1_state.params, f)
            with open(f'{model_dir}/hover_sac_critic2_params_{timestamp}.pkl', 'wb') as f:
                pickle.dump(sac.critic2_state.params, f)
            with open(f'{model_dir}/hover_sac_target_critic1_params_{timestamp}.pkl', 'wb') as f:
                pickle.dump(sac.target_critic1_state.params, f)
            with open(f'{model_dir}/hover_sac_target_critic2_params_{timestamp}.pkl', 'wb') as f:
                pickle.dump(sac.target_critic2_state.params, f)
                
if __name__ == "__main__":
    # Example usage
    set_logging_level('DEBUG')
    run()