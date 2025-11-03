import os
import random
import numpy as np
import time
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper 
import matplotlib.pyplot as plt_ex

ALREADY = 500     # episodes already trained
MORE = 100        # additional episodes to train
EPISODES = ALREADY + MORE

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.998
        self.learning_rate = 1e-4  # smaller LR for fine-tuning
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # store IMMEDIATE reward (not cumulative)
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return int(np.argmax(act_values[0]))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def save_full(self, path):
        self.model.save(path)

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    env = Environment("../Simulation2d/world/test")
    # env.set_mode(Mode.PAIR_ALL, terminate_at_end=True)
    # env.set_mode(Mode.ALL_RANDOM, terminate_at_end=False)
    env.use_observation_rotation_size(True)
    env.set_observation_rotation_size(128)

    state_size = env.observation_size()
    action_size = action_mapper.ACTION_SIZE
    agent = DQNAgent(state_size, action_size)

    # ---- resume from pretrained weights ----
    agent.load("weights/500_runs_weight.h5")   # path to your pretrained weights
    agent.epsilon = 0.1                          # small exploration for continued learning

    done = False
    batch_size = 48
    env.activate_visuals(True)

    agent_epsilons = []
    print("RESUME DQN TRAINING")

    for e in range(ALREADY, EPISODES):
        visualize = (e % 50 == 0)
        reward_sum = 0.0

        # robust reset
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        state = np.reshape(state, (1, state_size))

        for iteration in range(100):
            action = agent.act(state)
            linear, angular = action_mapper.map_action(action)

            # robust step
            step_out = env.step(linear, angular, 20)
            if isinstance(step_out, (tuple, list)) and len(step_out) >= 4:
                next_state, reward, done, _ = step_out[:4]
            else:
                next_state, reward, done = step_out

            next_state = np.reshape(next_state, (1, state_size))
            reward_sum += float(reward)

            # store immediate reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if visualize:
                env.visualize()

            if done:
                agent_epsilons.append(float(agent.epsilon))
                print(f"episode: {e}/{EPISODES}, score: {reward_sum:.3f}, e: {agent.epsilon:.3f}, steps:{iteration+1}")
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 50 == 0:
            plt_ex.plot(np.array(agent_epsilons))
            plt_ex.xlabel("Checkpoint index")
            plt_ex.ylabel("Agent epsilon")
            plt_ex.title("Agent's Epsilon (resume)")
            plt_ex.savefig(f"figs/after_{e}_episodes.png")
            plt_ex.clf()

            agent.save(f"weights/weight_after_{e}_episodes.h5")
            # optionally also save full model:
            # agent.save_full(f"weights2/model_after_{e}_episodes.keras")

    print("DQN Training (Resumed) Done")
    print(f"last: episode {e}/{EPISODES}, score {reward_sum:.3f}, e {agent.epsilon:.3f}, steps {iteration+1}")

    # final saves
    plt_ex.plot(np.array(agent_epsilons))
    plt_ex.xlabel("Checkpoint index")
    plt_ex.ylabel("Agent epsilon")
    plt_ex.title("Agent's Epsilon (final)")
    plt_ex.savefig("figs/final.png")
    agent.save("weights/weight_final_600.h5")
    # agent.save_full("weights2/model_final.keras")
