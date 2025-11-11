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

import matplotlib.pyplot as plt_ex

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




EPISODES = 50

class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
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

def manual_control():
    """
    Let the human control the agent manually.
    Press keys to control linear and angular velocity.
    """
    print("\nManual control mode (use W/A/S/D keys, p to pause/quit episode):")
    print("  W = forward, A = turn left, D = turn right")

    key = input("Enter action (W/A/D/Q/E): ").strip().lower()
    if key == 'w':
        linear, angular = 0.6, 0.0
    elif key == 'a':
        linear, angular = 0.5, 0.5
    elif key == 'd':
        linear, angular = 0.5, -0.5
    elif key == 'q':
        linear, angular = 0.3, 1.25
    elif key == 'e':
        linear, angular = 0.3, -1.25
    elif key == 'p':
        return None, None, True  # end episode manually
    else:
        print("Invalid input! Skipping...")
        linear, angular = 0.0, 0.0

    return linear, angular, False



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

    rewards_csv = Path("logs/rewards_with_human.csv")

    plot_model(
        agent.model,
        to_file="figs/dqn_model.png",
        show_shapes=True,         # show tensor shapes
        show_layer_names=True,    # show layer names
        expand_nested=False,
        dpi=200,
        rankdir="LR"              # "TB" top->bottom, or "LR" left->right
    )


    done = False
    batch_size = 48
    env.activate_visuals(True)

    agent_scores = []
    print("START DQN")


    for e in range(EPISODES):
        visualize = (e % 5 == 0 and e != 0)  
        reward_sum = 0.0

        state, _, _, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        # --- show the initial frame in manual episodes ---
        if e < 5:
            env.visualize()

        print(f"Episode {e} -> {'MANUAL' if e < 5 else 'AUTONOMOUS'}")

        for iteration in range(100):

            if e < 5:
                # --------- MANUAL MODE ----------
                linear, angular, manual_done = manual_control()
                if manual_done:
                    done = True
                    # still log & break like a normal done
                    agent_scores.append(float(reward_sum))
                    print(f"episode: {e}/{EPISODES}, score: {reward_sum}, e: {agent.epsilon:.2f} iteration:{iteration}")
                    append_reward(rewards_csv, e, reward_sum, agent.epsilon)
                    break

                # if you need an action index for replay, try to reverse-map; fallback to 0
                if hasattr(action_mapper, "reverse_map"):
                    action = action_mapper.reverse_map(linear, angular)
                else:
                    action = 0  # placeholder index if no reverse map exists

                next_state, reward, done, _ = env.step(linear, angular, 20)
                # ----- visualize EVERY STEP in manual mode -----
                env.visualize()

            else:
                # --------- AUTONOMOUS DQN ----------
                action = agent.act(state)
                linear, angular = action_mapper.map_action(action)
                next_state, reward, done, _ = env.step(linear, angular, 20)

                if visualize:
                    env.visualize()

            next_state = np.reshape(next_state, [1, state_size])
            reward_sum += reward

            # IMPORTANT: store the per-step reward (not reward_sum)
            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                agent_scores.append(float(reward_sum))
                print("episode: {}/{}, score: {}, e: {:.2f} iteration:{}"
                    .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
                append_reward(rewards_csv, e, reward_sum, agent.epsilon)
                break

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if e % 100 == 0 and e != 0:
            plt_ex.plot(np.array(agent_scores))
            plt_ex.xlabel("X-axis Label")
            plt_ex.ylabel("Agents Epsilons")
            plt_ex.title("Agent's Epsilons")
            plt_ex.savefig("figs/After {} episodes.png".format(e))
            agent.save("weights/{}_runs_weight_after{}_episodes.h5".format(EPISODES, e))



    print("DQN Done")
    print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                    .format(e, EPISODES, reward_sum, agent.epsilon, iteration))

    print(agent_scores)



    plt_ex.plot(np.array(agent_scores))


    plt_ex.xlabel("X-axis Label")
    plt_ex.ylabel("Agents Epsilons")
    plt_ex.title("Agent's Epsilons")

    plt_ex.savefig("figs/final.png")
    agent.save("weights/{}_runs_weight_final_epsilon_{}.h5".format(EPISODES, agent.epsilon))
            
            