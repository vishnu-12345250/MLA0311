import numpy as np
import random

# -----------------------------
# RTS Environment
# -----------------------------
class RTSEnvironment:
    def reset(self):
        self.state = np.array([50.0, 5.0, 40.0])  # resources, units, enemy
        return self.state

    def step(self, action):
        gather, build, attack = action

        # Apply action effects
        self.state[0] += gather * 4
        self.state[1] += build * 2
        self.state[2] -= attack * self.state[1]

        # Reward function
        reward = gather + build + attack * 5

        # Terminal condition
        done = self.state[2] <= 0

        return self.state, reward, done


# -----------------------------
# Actor (Policy)
# -----------------------------
class Actor:
    def predict(self, state):
        # Deterministic policy (DDPG-style)
        gather = min(1, state[0] / 100)
        build = min(1, state[1] / 20)
        attack = min(1, state[1] / max(state[2], 1))
        return np.array([gather, build, attack])


# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, size=1000):
        self.buffer = []
        self.size = size

    def add(self, experience):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size=5):
        return random.sample
