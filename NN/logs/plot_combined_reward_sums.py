
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- Config ----------
CSV_PATH = "rewards_combined.csv"
OUT_PATH = Path("rewards_vs_episode.png")

# ---------- Load Data ----------
df = pd.read_csv(CSV_PATH, sep=',', skipinitialspace=True)

# ---------- Plot ----------
plt.figure(figsize=(10, 6))
plt.plot(df["episode"], df["reward_sum"], label="Reward (Agent)", color="tab:blue", linewidth=2)
plt.plot(df["episode"], df["reward_sum_human"], label="Reward (Human)", color="tab:orange", linewidth=2)

# ---------- Styling ----------
plt.title("Agent vs Human Rewards per Episode", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# ---------- Save and Show ----------
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
plt.close()

print(f"Plot saved as {OUT_PATH.resolve()}")
