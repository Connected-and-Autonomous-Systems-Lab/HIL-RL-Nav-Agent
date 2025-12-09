#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A3C-style TF2/Keras navigation agent using the existing GA3C Environment.

- Uses ga3c.Environment.Environment as the environment wrapper
- Uses ga3c.Config for hyperparameters (discount, obs size, etc.)
- Network: Conv1D -> Conv1D -> Dense(256) -> policy(7) + value(1)
- Single-process, on-policy Advantage Actor-Critic (A2C-style)
"""

import os
import time
import argparse
from typing import List, Tuple
import csv
from datetime import datetime

import numpy as np
import tensorflow as tf

# Import your existing GA3C components
from ga3c.Config import Config
from ga3c.Environment import Environment as GA3CEnvironment


def build_actor_critic(
    obs_size: int,
    stacked_frames: int,
    num_actions: int,
) -> tf.keras.Model:
    """
    Build an Actor-Critic network similar to NetworkVP (1D convs + dense).
    Input shape matches GA3C: (OBSERVATION_SIZE, STACKED_FRAMES)
    """
    inputs = tf.keras.Input(
        shape=(obs_size, stacked_frames), name="state", dtype=tf.float32
    )

    # Conv1D layers (similar to NetworkVP: filter_size=9,16 and 5,32) 
    x = tf.keras.layers.Conv1D(
        filters=16,
        kernel_size=9,
        strides=5,
        activation="relu",
        padding="same",
        name="conv1",
    )(inputs)

    x = tf.keras.layers.Conv1D(
        filters=32,
        kernel_size=5,
        strides=3,
        activation="relu",
        padding="same",
        name="conv2",
    )(x)

    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)

    # Policy head: probabilities over actions
    policy = tf.keras.layers.Dense(
        num_actions, activation="softmax", name="policy"
    )(x)

    # Value head: scalar state-value
    value = tf.keras.layers.Dense(1, activation=None, name="value")(x)

    model = tf.keras.Model(inputs=inputs, outputs=[policy, value], name="A3CNav")
    return model


def compute_returns_and_advantages(
    rewards: List[float],
    dones: List[bool],
    values: List[float],
    next_value: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute discounted returns and advantages for a rollout.
    - rewards, dones, values: lists over time
    - next_value: bootstrap value (0 if terminal)
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)
    advantages = np.zeros(T, dtype=np.float32)

    # Start from bootstrap value
    G = next_value
    for t in reversed(range(T)):
        if dones[t]:
            G = 0.0
        G = rewards[t] + gamma * G
        returns[t] = G

    values = np.array(values, dtype=np.float32)
    advantages = returns - values
    return returns, advantages


@tf.function
def train_on_batch(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    states: tf.Tensor,         # (B, obs_size, stacked_frames)
    actions: tf.Tensor,        # (B,)
    returns: tf.Tensor,        # (B,)
    advantages: tf.Tensor,     # (B,)
    entropy_coef: float,
) -> tf.Tensor:
    """
    Single A2C-style update on one rollout batch.
    """
    with tf.GradientTape() as tape:
        policy, values = model(states, training=True)
        values = tf.squeeze(values, axis=-1)  # (B,)

        # Select log-probs of taken actions
        eps = 1e-10
        batch_indices = tf.range(tf.shape(policy)[0])
        indices = tf.stack([batch_indices, actions], axis=1)  # (B, 2)
        selected_probs = tf.gather_nd(policy, indices)
        log_probs = tf.math.log(tf.clip_by_value(selected_probs, eps, 1.0))

        # Entropy (for exploration)
        entropy = -tf.reduce_sum(
            policy * tf.math.log(tf.clip_by_value(policy, eps, 1.0)), axis=1
        )

        # Actor loss: policy gradient with advantage
        actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantages))

        # Critic loss: value regression
        critic_loss = tf.reduce_mean(tf.square(returns - values))

        # Entropy regularization
        entropy_loss = tf.reduce_mean(entropy)

        total_loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss


def run_episode(
    env: GA3CEnvironment,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    time_max: int,
    entropy_coef: float,
    obs_size: int,
    stacked_frames: int,
    reward_clip_min: float = None,
    reward_clip_max: float = None,
) -> Tuple[float, int]:
    """
    Run one episode and perform one or more A2C updates along the way.

    Returns:
        episode_reward, episode_length
    """
    # Reset environment (this will also pick new start/end nodes)
    env.reset()
    done = False

    episode_reward = 0.0
    episode_length = 0

    # Rollout buffers
    states_buf: List[np.ndarray] = []
    actions_buf: List[int] = []
    rewards_buf: List[float] = []
    dones_buf: List[bool] = []
    values_buf: List[float] = []

    while not done:
        # GA3C-style warmup: env.current_state is None until stacked frames ready
        if env.current_state is None:
            env.step(None)  # NOOP / warmup
            continue

        # State as (1, obs_size, stacked_frames)
        state = np.asarray(env.current_state, dtype=np.float32)
        state = state.reshape(1, 1210, 4)

        policy, value = model(state, training=False)
        policy = policy.numpy()[0]
        value = float(value.numpy()[0, 0])

        # Sample action from policy
        action = np.random.choice(len(policy), p=policy)

        # Take step in environment (GA3C-style: returns reward, done)
        reward, done = env.step(action)

        # Optional reward clipping
        if reward_clip_min is not None and reward_clip_max is not None:
            reward = float(np.clip(reward, reward_clip_min, reward_clip_max))

        episode_reward += reward
        episode_length += 1

        # Store transition
        states_buf.append(state[0])  # remove batch dim
        actions_buf.append(action)
        rewards_buf.append(reward)
        dones_buf.append(done)
        values_buf.append(value)

        # If we hit time_max or the episode ended, update network
        if len(states_buf) >= time_max or done:
            # Bootstrap from last state if not done
            if not done and env.current_state is not None:
                next_state = np.asarray(env.current_state, dtype=np.float32)
                next_state = next_state.reshape(1, obs_size, stacked_frames)
                _, next_val = model(next_state, training=False)
                next_value = float(next_val.numpy()[0, 0])
            else:
                next_value = 0.0

            # Compute returns & advantages
            returns_np, adv_np = compute_returns_and_advantages(
                rewards_buf,
                dones_buf,
                values_buf,
                next_value,
                gamma,
            )

            # Prepare tensors
            states_arr = np.stack(states_buf, axis=0).astype(np.float32)
            actions_arr = np.asarray(actions_buf, dtype=np.int32)

            states_tf = tf.convert_to_tensor(states_arr, dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(actions_arr, dtype=tf.int32)
            returns_tf = tf.convert_to_tensor(returns_np, dtype=tf.float32)
            adv_tf = tf.convert_to_tensor(adv_np, dtype=tf.float32)

            # One A2C-style update
            _ = train_on_batch(
                model,
                optimizer,
                states_tf,
                actions_tf,
                returns_tf,
                adv_tf,
                entropy_coef,
            )

            # Clear buffers but keep episode running
            states_buf.clear()
            actions_buf.clear()
            rewards_buf.clear()
            dones_buf.clear()
            values_buf.clear()

    return episode_reward, episode_length


def main():
    parser = argparse.ArgumentParser(
        description="A3C-style TF2 navigation agent using GA3C Environment"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Number of training episodes (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides Config.LEARNING_RATE_START)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=None,
        help="Entropy coefficient (overrides Config.BETA_START)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save model every N episodes (default: 500)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="a3c_checkpoints",
        help="Directory to save model checkpoints and CSV log",
    )
    args = parser.parse_args()

    # Hyperparameters from Config, with CLI overrides
    obs_size = Config.OBSERVATION_SIZE
    stacked_frames = Config.STACKED_FRAMES
    gamma = getattr(Config, "DISCOUNT", 0.99)
    time_max = getattr(Config, "TIME_MAX", 20)
    lr = args.lr if args.lr is not None else getattr(Config, "LEARNING_RATE_START", 1e-4)
    entropy_coef = (
        args.entropy_coef
        if args.entropy_coef is not None
        else getattr(Config, "BETA_START", 0.01)
    )

    reward_clip_min = getattr(Config, "REWARD_MIN", None)
    reward_clip_max = getattr(Config, "REWARD_MAX", None)

    # Create environment (use agent_id=0)
    env = GA3CEnvironment(0)
    num_actions = env.get_num_actions()

    print("A3C TF2 Navigation")
    print(f"  obs_size       = {obs_size}")
    print(f"  stacked_frames = {stacked_frames}")
    print(f"  num_actions    = {num_actions}")
    print(f"  gamma          = {gamma}")
    print(f"  time_max       = {time_max}")
    print(f"  lr             = {lr}")
    print(f"  entropy_coef   = {entropy_coef}")

    # Build model & optimizer
    model = build_actor_critic(obs_size, stacked_frames, num_actions)

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=lr,
        rho=getattr(Config, "RMSPROP_DECAY", 0.99),
        momentum=getattr(Config, "RMSPROP_MOMENTUM", 0.0),
        epsilon=getattr(Config, "RMSPROP_EPSILON", 1e-10),
    )

    # Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- CSV logging setup ----------
    csv_path = os.path.join(args.outdir, "training_log.csv")
    csv_exists = os.path.isfile(csv_path)

    if not csv_exists:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "reward_sum",
                    "episode_length",
                    "moving_avg_reward",
                    "best_reward",
                    "timestamp",
                ]
            )
    # --------------------------------------

    # Training loop
    best_reward = -1e9
    reward_moving_avg = None

    start_time = time.time()
    for ep in range(1, args.episodes + 1):
        ep_reward, ep_len = run_episode(
            env=env,
            model=model,
            optimizer=optimizer,
            gamma=gamma,
            time_max=time_max,
            entropy_coef=entropy_coef,
            obs_size=obs_size,
            stacked_frames=stacked_frames,
            reward_clip_min=reward_clip_min,
            reward_clip_max=reward_clip_max,
        )

        if reward_moving_avg is None:
            reward_moving_avg = ep_reward
        else:
            reward_moving_avg = 0.99 * reward_moving_avg + 0.01 * ep_reward

        if ep_reward > best_reward:
            best_reward = ep_reward

        # Print every 10 episodes (you can change this to 1 if you want every episode)
        if ep % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"[Episode {ep:6d}] "
                f"Reward: {ep_reward:8.3f}  "
                f"Len: {ep_len:5d}  "
                f"AvgR: {reward_moving_avg:8.3f}  "
                f"BestR: {best_reward:8.3f}  "
                f"Time: {elapsed:7.1f}s"
            )

        # ---------- Append one row to CSV per episode ----------
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    ep,
                    ep_reward,
                    ep_len,
                    float(reward_moving_avg),
                    best_reward,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ]
            )
        # -------------------------------------------------------

        # Periodic checkpoint
        if ep % args.save_every == 0:
            ckpt_path = os.path.join(args.outdir, f"a3c_nav_ep{ep:06d}.weights.h5")
            model.save_weights(ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = os.path.join(args.outdir, f"a3c_nav_final_ep{args.episodes:06d}.weights.h5")
    model.save_weights(ckpt_path)
    print(f"Training finished. Final model saved to: {ckpt_path}")
    print(f"CSV log saved to: {csv_path}")



if __name__ == "__main__":
    main()
