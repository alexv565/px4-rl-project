import gym
import numpy as np
from gym import spaces


class SimDroneEnv(gym.Env):
    """
    Simple 1D altitude control sim.
    State: [altitude, vertical_velocity]
    Action: continuous thrust in [-1, 1]
    """

    def __init__(self, target_altitude=5.0, max_episode_steps=250):
        super().__init__()
        self.target_altitude = target_altitude
        self.max_episode_steps = max_episode_steps

        # State: altitude (m), vertical_velocity (m/s)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -10.0], dtype=np.float32),
            high=np.array([20.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: thrust delta [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.dt = 0.1  # time step
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.altitude = 0.0
        self.velocity = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.altitude, self.velocity], dtype=np.float32)

    def step(self, action):
        thrust = float(np.clip(action[0], -1.0, 1.0))

        # Very simple physics-ish update
        acc = thrust * 5.0 - 9.81 * 0.1  # fake thrust vs gravity
        self.velocity += acc * self.dt
        self.altitude += self.velocity * self.dt
        self.altitude = max(0.0, self.altitude)

        self.step_count += 1

        # Reward: penalize distance from target, small velocity penalty
        alt_error = self.altitude - self.target_altitude
        reward = - (alt_error ** 2) - 0.1 * (self.velocity ** 2)

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info = {
            "altitude": self.altitude,
            "velocity": self.velocity,
        }

        return self._get_obs(), reward, terminated, truncated, info
