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

EPISODES = 5 

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 0.0            # <<— evaluation: no exploration
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(2048, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def load(self, name):
        self.model.load_weights(name)

    def act(self, state):
        # greedy action only
        act_values = self.model.predict(state, verbose=0)
        return int(np.argmax(act_values[0]))

if __name__ == "__main__":
    
    env = Environment("../Simulation2d/world/test")
    # env.set_mode(Mode.PAIR_ALL, terminate_at_end=True)
    # env.set_mode(Mode.ALL_RANDOM, terminate_at_end=False)
    env.use_observation_rotation_size(True)
    env.set_observation_rotation_size(128)

    state_size = env.observation_size()
    action_size = action_mapper.ACTION_SIZE

    agent = DQNAgent(state_size, action_size)
    agent.load("weights_done/500_runs_weight.h5")
    agent.epsilon = 0.0  

    env.activate_visuals(True)

    print("START DQN EVAL")
    scores = []

    for e in range(EPISODES):
        
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, (list, tuple)) else reset_out
        state = np.reshape(state, (1, state_size))

        reward_sum = 0.0
        done = False

        
        visualize = True  #  (e % 5 == 0)

        for iteration in range(150):  
        # while True:
            action = agent.act(state)
            linear, angular = action_mapper.map_action(action)

            step_out = env.step(linear, angular, 20)
            # handle step return shape similarly
            if isinstance(step_out, (list, tuple)) and len(step_out) >= 4:
                next_state, reward, done, _ = step_out[:4]
            else:
                next_state, reward, done = step_out  

            next_state = np.reshape(next_state, (1, state_size))
            reward_sum += float(reward)
            state = next_state

            if visualize:
                env.visualize()

            if done:
                break

        scores.append(reward_sum)
        print(f"episode: {e+1}/{EPISODES}, score: {reward_sum:.3f}, steps: {iteration+1}")

    avg = np.mean(scores) if scores else 0.0
    print("DQN Eval Done")
    print(f"Average score over {EPISODES} episodes: {avg:.3f}")
