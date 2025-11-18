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

from environment.dynamic_lidar_map import DynamicLidarMap


# # Prevent Windows from sleeping while this script runs
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

    def evaluate_model(self, batch_size):
        """
        Evaluate the model's performance using Mean Squared Error (MSE)
        over a random minibatch from memory.

        :param batch_size: Number of samples to evaluate.
        :return: Mean Squared Error (float)
        """
        # If not enough samples, return None
        if len(self.memory) < batch_size:
            return None

        minibatch = random.sample(self.memory, batch_size)
        mse_list = []

        for state, action, reward, next_state, done in minibatch:
            # Compute target value (same logic as in replay)
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            # Predicted Q-values for current state
            predicted_q = self.model.predict(state, verbose=0)[0][action]

            # MSE for this sample
            mse = (target - predicted_q) ** 2
            mse_list.append(mse)

        # Return mean MSE
        mean_mse = np.mean(mse_list)
        print(f"Evaluation MSE: {mean_mse:.6f}")
        return mean_mse




def append_reward(csv_path: Path, episode: int, reward_sum: float, epsilon: float) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["episode", "reward_sum", "epsilon",  "timestamp"])
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

    # ---- Dynamic LiDAR map (global, human-limited knowledge) ----
    dynamic_map = DynamicLidarMap(width_m=20.0, height_m=20.0, resolution=0.1)
    dynamic_map.clear_global_map()  # start with completely unknown map


    # ---- Live visualization for dynamic map ----
    plt_ex.ion()  # turn on interactive mode

    fig_map, ax_map = plt_ex.subplots()
    im_map = ax_map.imshow(dynamic_map.grid,
                           origin="lower",
                           vmin=-1, vmax=1)
    cbar = fig_map.colorbar(im_map, ax=ax_map)
    cbar.set_label("Occupancy (-1 unknown, 0 free, 1 occupied)")
    ax_map.set_title("Dynamic LiDAR Map (Episode 0)")
    plt_ex.tight_layout()
    plt_ex.show(block=False)


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
    agent_mses= []
    print("START DQN")



    for e in range(EPISODES):

        

        visualize = (e % 2 == 0 and e != 0)

        reward_sum = 0

        state, _, _, info = env.reset()
        state = np.reshape(state, [1, state_size])

        # Reset per-episode visited flags in dynamic map
        dynamic_map.reset_episode()

        # Optional: use initial pose to mark first visited cell
        init_robot_x = info["robot_x"]
        init_robot_y = info["robot_y"]
        dynamic_map.reward_for_step(init_robot_x, init_robot_y)


        for iteration in range(100):
            # 1) choose action
            action = agent.act(state)

            linear, angular = action_mapper.map_action(action)

            # 2) step environment and get info dict
            next_state, env_reward, done, info = env.step(linear, angular, 20)
            next_state = np.reshape(next_state, [1, state_size])

            # 3) extract pose + lidar from info
            robot_x = info["robot_x"]
            robot_y = info["robot_y"]
            robot_theta = info["robot_orientation"]

            ranges = info["lidar_ranges"]
            angle_min = info["lidar_angle_min"]
            angle_increment = info["lidar_angle_increment"]
            max_range = info["lidar_max_range"]

            # 4) update dynamic map from LiDAR
            dynamic_map.update_from_scan(
                robot_x, robot_y, robot_theta,
                ranges, angle_min, angle_increment, max_range
            )

            # 5) get exploration / revisit reward ONLY from limited map
            exploration_reward, revisit_penalty = dynamic_map.reward_for_step(
                robot_x, robot_y,
                exploration_reward_value=0.5,
                revisit_penalty_value=-0.1
            )

            # 6) total shaped reward for this step
            step_reward = env_reward + exploration_reward + revisit_penalty
            reward_sum += step_reward

            # 7) store transition with *step-level* reward
            agent.remember(state, action, step_reward, next_state, done)
            state = next_state

            if visualize:
                env.visualize()
                #time.sleep(1.0)

                # ---- Update dynamic map live ----
                im_map.set_data(dynamic_map.grid)
                ax_map.set_title(f"Dynamic LiDAR Map - Ep {e}, Step {iteration}")
                fig_map.canvas.draw()
                fig_map.canvas.flush_events()
                # Small pause to let GUI update (non-blocking)
                plt_ex.pause(0.001)
                

            if done:
                agent_scores.append(float(reward_sum))
                print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                    .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)      # Training the model
            append_reward(LOG_FILE, e, reward_sum, agent.epsilon)


        if e % 100 == 0 and e != 0:
            # agent.save("./save/dqn" + str(e) + ".h5")
            plt_ex.plot(np.array(agent_scores))

            plt_ex.xlabel("X-axis Label")
            plt_ex.ylabel("Agents Epsilons")
            plt_ex.title("Agent's Epsilons")

            plt_ex.savefig("figs/After {} episodes.png".format(e))
            mse = agent.evaluate_model(batch_size)
            agent.save("weights/{}_{}_runs_weight_after{}_episodes_with_revisit_penalty_mse_{}.h5".format(STARTING_TIME,EPISODES,e, mse))


    print("DQN Done")
    print("episode: {}/{}, score: {}, e: {:.2} iteration:{}"
                    .format(e, EPISODES, reward_sum, agent.epsilon, iteration))

    print(agent_scores)



    plt_ex.plot(np.array(agent_scores))


    plt_ex.xlabel("X-axis Label")
    plt_ex.ylabel("Agents Epsilons")
    plt_ex.title("Agent's Epsilons")

    plt_ex.savefig("figs/final.png")
    agent.save("weights/{}_{}_runs_weight_final_epsilon_{}_with_revisit_penalty.h5".format(STARTING_TIME,EPISODES, agent.epsilon))
            
            