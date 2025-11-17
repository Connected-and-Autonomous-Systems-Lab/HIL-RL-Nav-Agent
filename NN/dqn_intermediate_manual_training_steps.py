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


EPISODES = 100

# --- New constants for manual/auto alternation ---
MANUAL_EPISODES = 5      # first 5 episodes use manual/auto chunks
MANUAL_CHUNK = 20        # 20 manual steps
AUTO_CHUNK = 20          # 20 auto steps
MAX_STEPS_PER_EP = 100   # maximum steps per episode


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
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
    print("\nManual control mode (use W/A/D/Q/E keys, P to end episode):")
    print("  W = forward, A = soft left, D = soft right")
    print("  Q = hard left, E = hard right, P = end episode")

    key = input("Enter action (W/A/D/Q/E/P): ").strip().lower()
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


def is_mixed_mode_episode(e: int) -> bool:
    """
    Use MANUAL/AUTO chunks for:
      - First MANUAL_EPISODES (1..MANUAL_EPISODES in human counting)
      - Every 10th episode (10, 20, 30, ...) in human counting
    """
    human_episode = e + 1
    return (human_episode <= MANUAL_EPISODES) or (human_episode % 10 == 0)


if __name__ == "__main__":
    env = Environment("../Simulation2d/world/test")
    # env.set_mode(Mode.PAIR_ALL, terminate_at_end=True)
    # env.set_mode(Mode.ALL_RANDOM, terminate_at_end=False)
    env.use_observation_rotation_size(True)
    # env.set_cluster_size(10)
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

    batch_size = 48
    env.activate_visuals(True)

    agent_scores = []
    print("START DQN")

    for e in range(EPISODES):
        visualize = (e % 5 == 0 and e != 0)
        reward_sum = 0.0
        done = False

        state, _, _, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        # Decide if this episode uses manual/auto chunks
        mixed_mode = is_mixed_mode_episode(e)

        # show initial frame for mixed episodes
        if mixed_mode:
            env.visualize()

        print(f"Episode {e} -> {'MANUAL/AUTO CHUNKS' if mixed_mode else 'AUTONOMOUS'}")

        # mode & chunk setup
        mode = 'manual' if mixed_mode else 'auto'
        steps_in_chunk = 0

        for iteration in range(MAX_STEPS_PER_EP):

            if mode == 'manual':
                # --------- MANUAL STEP ----------
                linear, angular, manual_done = manual_control()
                if manual_done:
                    done = True
                    agent_scores.append(float(reward_sum))
                    print(f"episode: {e}/{EPISODES}, score: {reward_sum}, e: {agent.epsilon:.2f} iteration:{iteration}")
                    append_reward(rewards_csv, e, reward_sum, agent.epsilon)
                    break

                if hasattr(action_mapper, "reverse_map"):
                    action = action_mapper.reverse_map(linear, angular)
                else:
                    action = 0  # fallback if no reverse_map exists

                next_state, reward, done, _ = env.step(linear, angular, 20)
                env.visualize()  # visualize every manual step

            else:
                # --------- AUTONOMOUS DQN STEP ----------
                action = agent.act(state)
                linear, angular = action_mapper.map_action(action)
                next_state, reward, done, _ = env.step(linear, angular, 20)

                # visualize during mixed episodes or according to your old flag
                if mixed_mode:
                    env.visualize()
                else:
                    if visualize:
                        env.visualize()

            # --- common bookkeeping for both modes ---
            next_state = np.reshape(next_state, [1, state_size])
            reward_sum += reward

            # store per-step reward (not cumulative)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                agent_scores.append(float(reward_sum))
                print("episode: {}/{}, score: {}, e: {:.2f} iteration:{}"
                      .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
                append_reward(rewards_csv, e, reward_sum, agent.epsilon)
                break

            # --- Chunk switching logic (only for mixed episodes) ---
            if mixed_mode:
                steps_in_chunk += 1
                if mode == 'manual' and steps_in_chunk >= MANUAL_CHUNK:
                    mode = 'auto'
                    steps_in_chunk = 0
                    print(f"[Episode {e}] Switching to AUTO chunk")
                elif mode == 'auto' and steps_in_chunk >= AUTO_CHUNK:
                    mode = 'manual'
                    steps_in_chunk = 0
                    print(f"[Episode {e}] Switching to MANUAL chunk")

        # --- Training after each episode ---
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # --- Periodic saving/plotting ---
        if e % 100 == 0 and e != 0:
            plt_ex.plot(np.array(agent_scores))
            plt_ex.xlabel("X-axis Label")
            plt_ex.ylabel("Agents Epsilons")
            plt_ex.title("Agent's Epsilons")
            plt_ex.savefig("figs/After {} episodes.png".format(e))
            agent.save("weights/{}_runs_weight_after{}_episodes.h5".format(EPISODES, e))

    print("DQN Done")
    print("episode: {}/{}, score: {}, e: {:.2f} iteration:{}"
          .format(e, EPISODES, reward_sum, agent.epsilon, iteration))

    print(agent_scores)

    plt_ex.plot(np.array(agent_scores))
    plt_ex.xlabel("X-axis Label")
    plt_ex.ylabel("Agents Epsilons")
    plt_ex.title("Agent's Epsilons")
    plt_ex.savefig("figs/final.png")
    agent.save("weights/{}_runs_weight_final_epsilon_{}.h5".format(EPISODES, agent.epsilon))
