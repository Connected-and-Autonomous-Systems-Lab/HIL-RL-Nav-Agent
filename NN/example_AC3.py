# -*- coding: utf-8 -*-
import random
import numpy as np
import time
from collections import deque

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

EPISODES = 10000

class ACAgent:
    """
    Actor-Critic Agent with Experience Replay, adapting A3C concepts.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0  # Exploration rate (used for initial exploration)
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.beta = 0.01     # Entropy regularization coefficient

        self.model = self._build_model()
        self.optimizer = Adam(lr=self.learning_rate)
        # Compile the custom training step into a TensorFlow function for performance
        self.train_step_fn = self._create_train_step()

    def _build_model(self):
        """
        Builds the dual-head Actor-Critic network using Keras Functional API.
        The network has a shared backbone, an Actor head (policy), and a Critic head (value).
        """
        # Shared Input
        state_input = Input(shape=(self.state_size,), name='state_input')
        
        # Shared Backbone
        shared = Dense(2048, activation='relu')(state_input)
        shared = Dense(512, activation='relu')(shared)
        shared = Dense(256, activation='relu')(shared)
        
        # Actor Head (Policy: outputs probability distribution over actions)
        policy_output = Dense(self.action_size, activation='softmax', name='policy_output')(shared)
        
        # Critic Head (Value: outputs single scalar V(s))
        value_output = Dense(1, activation='linear', name='value_output')(shared)
        
        # The model returns both policy (probabilities) and value (scalar V(s))
        model = Model(inputs=state_input, outputs=[policy_output, value_output])
        return model

    def _create_train_step(self):
        """
        Defines the custom Actor-Critic training step using TensorFlow's GradientTape.
        """
        @tf.function
        def train_step(states, target_returns, action_masks):
            with tf.GradientTape() as tape:
                # Forward pass: get current policy and value predictions
                policy_pred, value_pred = self.model(states, training=True)
                value_pred = K.squeeze(value_pred, axis=1) # V(s) is 1D

                # 1. Critic Loss (Value Loss)
                # L_V = 0.5 * (R_t - V(s))^2
                value_loss = 0.5 * K.square(target_returns - value_pred)
                value_loss = K.mean(value_loss)
                
                # 2. Policy Loss (Actor Loss)
                
                # Advantage: A(s, a) = R_t - V(s)
                # The Critic's prediction V(s) is detached from the gradient calculation for the Actor loss.
                advantage = target_returns - K.stop_gradient(value_pred)
                
                # Probability of selected action: \pi(a|s)
                action_prob = K.sum(policy_pred * action_masks, axis=-1)

                # Log probability of selected action: log(\pi(a|s))
                log_action_prob = K.log(K.clip(action_prob, 1e-10, 1.0))

                # Policy Gradient Term: log(\pi(a|s)) * A(s, a)
                policy_gradient_term = log_action_prob * advantage
                
                # Entropy Term: \beta * H(\pi(s))
                # H(\pi(s)) = -\sum_a \pi(a|s) \log(\pi(a|s))
                entropy = K.sum(policy_pred * K.log(K.clip(policy_pred, 1e-10, 1.0)), axis=-1)
                
                # Total Policy Loss: - (Policy Gradient Term + \beta * Entropy Term)
                policy_loss = - K.mean(policy_gradient_term + self.beta * entropy)
                
                # 3. Total Loss: Sum of Actor and Critic losses
                total_loss = policy_loss + value_loss

            # Apply gradients
            trainable_vars = self.model.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            return total_loss, policy_loss, value_loss

        return train_step

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy exploration: random action if within epsilon threshold
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Predict policy (probabilities) and value (V(s))
        policy, _ = self.model.predict(state)
        policy = policy[0]
        
        # Sample action from the policy distribution (Actor)
        action = np.random.choice(self.action_size, p=policy)
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        # Extract and stack the state/next_state data
        states = np.array([exp[0][0] for exp in minibatch])
        next_states = np.array([exp[3][0] for exp in minibatch])
        
        # Predict V(s) for next states to calculate the Target Return R_t
        _, next_values = self.model.predict(next_states)
        
        # Prepare targets (R_t) and action masks
        target_returns = np.zeros(batch_size, dtype=np.float32)
        action_masks = np.zeros((batch_size, self.action_size), dtype=np.float32)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            # Target Return (R_t) calculation: R_t = r if done, else r + gamma * V(s')
            target = reward
            if not done:
                # Use the critic's prediction V(s') for the target
                target = (reward + self.gamma * next_values[i][0])
            
            target_returns[i] = target
            action_masks[i][action] = 1.0 # One-hot encode the action taken

        # Convert to TensorFlow Tensors
        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        target_returns_t = tf.convert_to_tensor(target_returns, dtype=tf.float32)
        action_masks_t = tf.convert_to_tensor(action_masks, dtype=tf.float32)

        # Execute the custom training step
        total_loss, policy_loss, value_loss = self.train_step_fn(
            states_t, target_returns_t, action_masks_t
        )

        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    # The setup below remains identical to your original DQN script
    env = Environment("../Simulation2d/world/test")
    env.use_observation_rotation_size(True)
    env.set_observation_rotation_size(128)

    state_size = env.observation_size()
    action_size = action_mapper.ACTION_SIZE
    # Initialize the new AC Agent
    agent = ACAgent(state_size, action_size) 
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 48

    print("START ACTOR-CRITIC (AC) WITH EXPERIENCE REPLAY")

    for e in range(EPISODES):

        visualize = (e % 5 == 0)

        reward_sum = 0

        state, _, _, _ = env.reset()

        # State must be reshaped for the Keras model input layer
        state = np.reshape(state, [1, state_size])

        for iteration in range(100):
            action = agent.act(state)

            linear, angular = action_mapper.map_action(action)

            next_state, reward, done, _ = env.step(linear, angular, 20)

            next_state = np.reshape(next_state, [1, state_size])

            reward_sum = reward_sum + reward

            agent.remember(state, action, reward_sum, next_state, done)
            state = next_state

            if visualize:
                env.visualize()

            if done:
                # The score printed here is the cumulative reward for the entire episode
                print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                      .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
                break
        
        # Train the model after the episode if enough memory is available
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            
        if e % 1000 == 0:
             agent.save("./weights/ac-replay" + str(e) + ".h5")