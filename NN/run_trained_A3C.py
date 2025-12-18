#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a trained A3C TF2 navigation agent (evaluation only).

- Loads weights saved by a3c_tf2_nav.py (build_actor_critic model)
- Uses ga3c.Environment.Environment for the nav environment
- Runs a fixed number of episodes and reports per-episode and average reward
"""

import numpy as np
import tensorflow as tf

from ga3c.Config import Config
from ga3c.Environment import Environment as GA3CEnvironment

# Import the same builder used in training
from a3c_tf2_nav import build_actor_critic


# ---------- HARD-CODED EVAL CONFIG ----------
WEIGHTS_PATH = "a3c_checkpoints/a3c_nav_ep000002.weights.h5"
EPISODES = 5
MAX_STEPS_PER_EPISODE = 150

# If False -> greedy argmax, if True -> sample from policy
STOCHASTIC_ACTIONS = False

# If True and env supports visualize(), show environment each step
VISUALIZE = True
# -------------------------------------------


def select_action_from_policy(policy: np.ndarray, stochastic: bool) -> int:
    """
    Choose an action from a policy distribution.

    If stochastic=False: greedy argmax
    If stochastic=True:  sample from the categorical distribution
    """
    if stochastic:
        return int(np.random.choice(len(policy), p=policy))
    else:
        return int(np.argmax(policy))


def main():
    # Hyperparameters (must match training setup)
    obs_size = Config.OBSERVATION_SIZE
    stacked_frames = Config.STACKED_FRAMES

    # Create GA3C environment
    env = GA3CEnvironment(0)
    num_actions = env.get_num_actions()

    print("A3C TF2 Navigation EVAL")
    print(f"  obs_size       = {obs_size}")
    print(f"  stacked_frames = {stacked_frames}")
    print(f"  num_actions    = {num_actions}")
    print(f"  episodes       = {EPISODES}")
    print(f"  max_steps      = {MAX_STEPS_PER_EPISODE}")
    print(f"  weights        = {WEIGHTS_PATH}")
    print(f"  stochastic     = {STOCHASTIC_ACTIONS}")
    print(f"  visualize      = {VISUALIZE}")

    # Build model exactly as in training and load weights
    model = build_actor_critic(obs_size, stacked_frames, num_actions)
    model.load_weights(WEIGHTS_PATH)
    print("Loaded weights from:", WEIGHTS_PATH)
    model.summary()

    scores = []

    for ep in range(1, EPISODES + 1):
        env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done and ep_len < MAX_STEPS_PER_EPISODE:
            # GA3C-style warmup: env.current_state is None until frames are stacked
            if env.current_state is None:
                env.step(None)
                continue

            # current_state shape: (obs_size, stacked_frames)
            state = np.asarray(env.current_state, dtype=np.float32)
            state = state.reshape(1, obs_size, stacked_frames)

            # Forward pass through actor-critic
            policy, value = model(state, training=False)
            policy = policy.numpy()[0]  # shape: (num_actions,)

            # Choose action
            action = select_action_from_policy(policy, STOCHASTIC_ACTIONS)

            # Step environment
            reward, done = env.step(action)

            ep_reward += float(reward)
            ep_len += 1

            if VISUALIZE:
                try:
                    env.visualize()
                except AttributeError:
                    pass

        scores.append(ep_reward)
        print(f"episode: {ep}/{EPISODES}, score: {ep_reward:.3f}, steps: {ep_len}")

    avg_score = float(np.mean(scores)) if scores else 0.0
    print("A3C Eval Done")
    print(f"Average score over {EPISODES} episodes: {avg_score:.3f}")


if __name__ == "__main__":
    main()
