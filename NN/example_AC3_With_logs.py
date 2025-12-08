# -*- coding: utf-8 -*-
import random
import numpy as np
import time
from collections import deque
import csv, os
from pathlib import Path
from datetime import datetime

# Use full tensorflow imports for the Functional API and GradientTape
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Assuming environment/environment.py and action_mapper are available/functional
from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper

# ----------------- CONFIG -----------------
EPISODES = 10000  # change if you want longer training

STARTING_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = Path(f"logs/{STARTING_TIME}_ac_rewards.csv")

# -------------- LOGGING UTILS --------------
def flush_log_buffer(csv_path: Path, buffer: list) -> None:
    """
    Write all buffered log rows to CSV and clear the buffer.
    Each row is [episode, reward_sum, epsilon, timestamp].
    """
    if not buffer:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            # w.writerow(["episode", "reward_sum", "epsilon", "timestamp"])
            w.writerow(["episode", "reward_sum", "epsilon"])
        for row in buffer:
            w.writerow(row)
    buffer.clear()


class ACAgent:
    """
    Actor-Critic Agent with Experience Replay, adapting A3C concepts.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate (used for initial exploration)
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.beta = 0.01     # Entropy regularization coefficient

        self.model = self._build_model()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.train_step_fn = self._create_train_step()

    def _build_model(self):
        """
        Builds the dual-head Actor-Critic network using Keras Functional API.
        The network has a shared backbone, an Actor head (policy), and a Critic head (value).
        """
        state_input = Input(shape=(self.state_size,), name='state_input')
        
        # Shared Backbone
        shared = Dense(2048, activation='relu')(state_input)
        shared = Dense(512, activation='relu')(shared)
        shared = Dense(256, activation='relu')(shared)
        
        # Actor Head (Policy)
        policy_output = Dense(self.action_size, activation='softmax',
                              name='policy_output')(shared)
        
        # Critic Head (Value)
        value_output = Dense(1, activation='linear',
                             name='value_output')(shared)
        
        model = Model(inputs=state_input, outputs=[policy_output, value_output])
        return model

    def _create_train_step(self):
        """
        Defines the custom Actor-Critic training step using TensorFlow's GradientTape.
        """
        @tf.function
        def train_step(states, target_returns, action_masks):
            with tf.GradientTape() as tape:
                policy_pred, value_pred = self.model(states, training=True)
                value_pred = K.squeeze(value_pred, axis=1)  # (batch,)

                # Critic loss
                value_loss = 0.5 * K.square(target_returns - value_pred)
                value_loss = K.mean(value_loss)
                
                # Advantage
                advantage = target_returns - K.stop_gradient(value_pred)
                
                # Prob of selected actions
                action_prob = K.sum(policy_pred * action_masks, axis=-1)
                log_action_prob = K.log(K.clip(action_prob, 1e-10, 1.0))

                # Policy gradient term
                policy_gradient_term = log_action_prob * advantage

                # Entropy (for exploration)
                entropy = K.sum(policy_pred * K.log(K.clip(policy_pred, 1e-10, 1.0)),
                                axis=-1)

                # Policy loss (negative because we maximize PG + entropy)
                policy_loss = -K.mean(policy_gradient_term + self.beta * entropy)

                total_loss = policy_loss + value_loss

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            return total_loss, policy_loss, value_loss

        return train_step

    def remember(self, state, action, reward, next_state, done):
        # NOTE: you are currently storing cumulative reward (reward_sum) here,
        # mirroring your DQN code.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        policy, _ = self.model.predict(state)
        policy = policy[0]
        action = np.random.choice(self.action_size, p=policy)
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([exp[0][0] for exp in minibatch])
        next_states = np.array([exp[3][0] for exp in minibatch])
        
        _, next_values = self.model.predict(next_states)
        
        target_returns = np.zeros(batch_size, dtype=np.float32)
        action_masks = np.zeros((batch_size, self.action_size), dtype=np.float32)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = reward
            if not done:
                target = reward + self.gamma * next_values[i][0]
            target_returns[i] = target
            action_masks[i][action] = 1.0

        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        target_returns_t = tf.convert_to_tensor(target_returns, dtype=tf.float32)
        action_masks_t = tf.convert_to_tensor(action_masks, dtype=tf.float32)

        self.train_step_fn(states_t, target_returns_t, action_masks_t)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        Path(os.path.dirname(name)).mkdir(parents=True, exist_ok=True)
        self.model.save_weights(name)


if __name__ == "__main__":
    env = Environment("../Simulation2d/world/test")
    env.use_observation_rotation_size(True)
    env.set_observation_rotation_size(128)

    state_size = env.observation_size()
    action_size = action_mapper.ACTION_SIZE

    agent = ACAgent(state_size, action_size)
    done = False
    batch_size = 48

    # Buffer to hold logs for 100-episode chunks
    log_buffer = []

    print("START ACTOR-CRITIC (AC) WITH EXPERIENCE REPLAY")

    for e in range(EPISODES):

        visualize = (e % 5 == 0)

        reward_sum = 0.0
        state, _, _, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        for iteration in range(100):
            action = agent.act(state)
            linear, angular = action_mapper.map_action(action)

            next_state, reward, done, _ = env.step(linear, angular, 20)
            next_state = np.reshape(next_state, [1, state_size])

            reward_sum += reward

            # Storing cumulative reward like in your DQN
            agent.remember(state, action, reward_sum, next_state, done)
            state = next_state

            if visualize:
                env.visualize()

            if done:
                print("episode: {}/{}, score: {}, e: {:.2f}, iteration:{}"
                      .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
                # Add this episode's log to the in-memory buffer only
                log_buffer.append([
                    e,
                    float(reward_sum),
                    float(agent.epsilon),
                    # datetime.now().isoformat(),
                ])
                break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        # Every 100 episodes, save weights AND flush logs to CSV
        if (e + 1) % 1000 == 0:
            weights_path = f"weights/{STARTING_TIME}_{EPISODES}_runs_ac_weight_after_{e+1}_episodes.h5"
            agent.save(weights_path)
            print(f"Saved weights to {weights_path}")

            # Flush log buffer to disk
            flush_log_buffer(LOG_FILE, log_buffer)
            print(f"Flushed logs for episodes up to {e+1} into {LOG_FILE}")

    # After all episodes, make sure any remaining logs are written
    flush_log_buffer(LOG_FILE, log_buffer)

    final_weights_path = f"weights/{STARTING_TIME}_ac_final_weights.h5"
    agent.save(final_weights_path)
    print(f"Training done. Final weights saved to {final_weights_path}")
