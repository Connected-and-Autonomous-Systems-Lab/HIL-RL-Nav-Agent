import random
import numpy as np
import time
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed
from keras.layers import SimpleRNN
from keras.optimizers import Adam

from environment.environment import Environment
from environment.environment_node_data import Mode
import action_mapper 

from tensorflow.keras.utils import plot_model

import csv, os
from pathlib import Path
from datetime import datetime


# Prevent Windows from sleeping while this script runs
# import ctypes, atexit

# ES_CONTINUOUS        = 0x80000000
# ES_SYSTEM_REQUIRED   = 0x00000001
# ES_DISPLAY_REQUIRED  = 0x00000002
# ES_AWAYMODE_REQUIRED = 0x00000040  # useful on AC power

# def _stay_awake():
#     ctypes.windll.kernel32.SetThreadExecutionState(
#         ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_AWAYMODE_REQUIRED
#     )

# def _allow_sleep():
#     ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# _stay_awake()
# atexit.register(_allow_sleep)




EPISODES = 100
STARTING_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FILE = Path("logs/{}_rewards.csv".format(STARTING_TIME))

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        # self.learning_rate = 0.001
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001

        self.model = self._build_model()

    

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(2048, input_dim=self.state_size, activation='relu'))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))


        
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                        np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def append_reward(csv_path: Path, episode: int, reward_sum: float, epsilon: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["episode", "reward_sum", "epsilon", "timestamp"])
        w.writerow([episode, float(reward_sum), float(epsilon), datetime.now().isoformat()])


if __name__ == "__main__":
    env = Environment("../Simulation2d/world/test")
    #env.set_mode(Mode.PAIR_ALL, terminate_at_end=True)
    #env.set_mode(Mode.ALL_RANDOM, terminate_at_end=False)
    env.use_observation_rotation_size(True)
    #env.set_cluster_size(10)
    env.set_observation_rotation_size(128)

    state_size = env.observation_size()
    action_size = action_mapper.ACTION_SIZE
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")




    done = False
    batch_size = 48
    env.activate_visuals(True)

    print("START DQN")



    for e in range(EPISODES):


        reward_sum = 0

        state, _, _, _ = env.reset()

        state = np.reshape(state, [1, state_size])
        print("initial statr: ", state)

        for iteration in range(100):
            action = agent.act(state)

            linear, angular = action_mapper.map_action(action)

            next_state, reward, done, _ = env.step(linear, angular, 20)

            next_state = np.reshape(next_state, [1, state_size])
            # print("reward = ", reward)

            reward_sum = reward_sum + reward

            agent.remember(state, action, reward_sum, next_state, done)
            state = next_state
                

            if done:
                print("episode: {}/{}, reward_sum: {}, e: {:.2} iteration:{}"
                    .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
            
                append_reward(LOG_FILE, e, reward_sum, agent.epsilon)
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 100 == 0 and e != 0:
            # agent.save("./save/dqn" + str(e) + ".h5")
            agent.save("weights/{}_{}_runs_weight_after{}_episodes.h5".format(STARTING_TIME, EPISODES,e))
            # agent.save("weights/500_runs_model_after{}_episodes.keras".format(e))


    print("DQN Done")
    print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                    .format(e, EPISODES, reward_sum, agent.epsilon, iteration))



    agent.save("weights/{}_final_weights.h5".format(STARTING_TIME))
            
            