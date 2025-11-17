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

# --- Constants for human reward-giving episodes ---
REWARD_EPISODES = 5        # first 5 episodes use human reward chunks
REWARD_CHUNK = 20          # every 20 steps, you give one reward multiplier
MAX_STEPS_PER_EP = 100     # maximum steps per episode


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
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
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


def is_reward_episode(e: int) -> bool:
    """
    Episodes where human gives chunk-level reward multipliers:
      - First REWARD_EPISODES (1..REWARD_EPISODES in human counting)
      - Every 10th episode (10, 20, 30, ...) in human counting
    """
    human_episode = e + 1
    return (human_episode <= REWARD_EPISODES) or (human_episode % 10 == 0)


def get_chunk_reward_from_human(ep: int, chunk_idx: int, steps_in_chunk: int) -> float:
    """
    Ask the human for a single reward multiplier for the last chunk of steps.
    This value will be multiplied with each step's environment reward.
    """
    while True:
        try:
            val = float(input(
                f"[Episode {ep}] Chunk {chunk_idx} ({steps_in_chunk} steps) "
                f"=> Enter human reward multiplier (e.g., -1, 0.5, 2): "
            ))
            return val
        except ValueError:
            print("Invalid input, please enter a numeric value (e.g., -1, 0.5, 2).")


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

        reward_mode = is_reward_episode(e)
        mode_str = "HUMAN REWARD EPISODE" if reward_mode else "AUTONOMOUS EPISODE"
        print(f"Episode {e} -> {mode_str}")

        # For reward episodes, we buffer transitions in chunks of REWARD_CHUNK
        # Store env_reward so we can multiply by human scalar later
        chunk_buffer = []   # list of (state, action, env_reward, next_state, done)
        steps_in_chunk = 0
        chunk_index = 0

        if reward_mode:
            env.visualize()

        for iteration in range(MAX_STEPS_PER_EP):
            # --------- AGENT ACTION (common) ----------
            action = agent.act(state)
            linear, angular = action_mapper.map_action(action)
            next_state, env_reward, done, _ = env.step(linear, angular, 20)
            next_state = np.reshape(next_state, [1, state_size])

            if reward_mode:
                # --------- HUMAN-REWARD EPISODE ----------
                env.visualize()
                print(
                    f"[Ep {e} Step {iteration}] "
                    f"action={action}, lin={linear:.2f}, ang={angular:.2f}, "
                    f"env_reward={env_reward:.3f}"
                )

                # buffer this transition (including env_reward)
                chunk_buffer.append((state, action, env_reward, next_state, done))
                steps_in_chunk += 1

                # when chunk full or episode finishes, ask for reward multiplier
                if steps_in_chunk >= REWARD_CHUNK or done:
                    chunk_index += 1
                    human_reward_mult = get_chunk_reward_from_human(
                        ep=e,
                        chunk_idx=chunk_index,
                        steps_in_chunk=steps_in_chunk
                    )

                    # multiply env_reward by the human reward multiplier
                    for s, a, r_env, ns, d in chunk_buffer:
                        final_reward = r_env * human_reward_mult
                        agent.remember(s, a, final_reward, ns, d)
                        reward_sum += final_reward

                    # reset chunk buffer
                    chunk_buffer = []
                    steps_in_chunk = 0

            else:
                # --------- PURE AUTONOMOUS EPISODE ----------
                if visualize:
                    env.visualize()

                # use environment reward directly
                agent.remember(state, action, env_reward, next_state, done)
                reward_sum += env_reward

            state = next_state

            if done:
                agent_scores.append(float(reward_sum))
                print("episode: {}/{}, score: {}, e: {:.2f} iteration:{}"
                      .format(e, EPISODES, reward_sum, agent.epsilon, iteration))
                append_reward(rewards_csv, e, reward_sum, agent.epsilon)
                break

        # --- Training after each episode ---
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        # --- Periodic saving/plotting ---
        if e % 100 == 0 and e != 0:
            plt_ex.plot(np.array(agent_scores))
            plt_ex.xlabel("Episode")
            plt_ex.ylabel("Episode Return")
            plt_ex.title("Agent's Returns Over Episodes")
            plt_ex.savefig("figs/After_{}_episodes.png".format(e))
            plt_ex.clf()
            agent.save("weights/{}_runs_weight_after_{}_episodes.h5".format(EPISODES, e))

    print("DQN Done")
    print("episode: {}/{}, score: {}, e: {:.2f} iteration:{}"
          .format(e, EPISODES, reward_sum, agent.epsilon, iteration))

    print(agent_scores)

    plt_ex.plot(np.array(agent_scores))
    plt_ex.xlabel("Episode")
    plt_ex.ylabel("Episode Return")
    plt_ex.title("Agent's Returns Over Episodes")
    plt_ex.savefig("figs/final.png")
    agent.save("weights/{}_runs_weight_final_epsilon_{}.h5".format(EPISODES, agent.epsilon))
