import asyncio
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mavsdk import System


class DroneEnv(gym.Env):
    """
    Minimal PX4 + MAVSDK Gym environment (v0).

    - Action space (for now): dummy 1D continuous action (no real control yet).
    - Observation: current altitude + step count.
    - Reward: how close we are to target altitude.
    - This is a skeleton to get the Gym loop working; we will add real control later.
    """

    metadata = {"render_modes": []}

    def __init__(self, target_altitude: float = 5.0, max_episode_steps: int = 200):
        super().__init__()

        # --- RL interface definitions ---
        # 1D action for now (placeholder: will become velocity or yaw commands later)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [altitude_m, normalized_step]
        self.observation_space = spaces.Box(
            low=np.array([-1000.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # --- PX4 / MAVSDK related ---
        self._drone: Optional[System] = None
        self._connected: bool = False

        # Single asyncio event loop for the entire env
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # --- Episode bookkeeping ---
        self._target_alt = float(target_altitude)
        self._max_steps = int(max_episode_steps)
        self._step_count = 0

    # --------- Helper coroutines ---------

    async def _ensure_connection(self) -> None:
        """Connect to PX4 SITL if not already connected."""
        if self._connected and self._drone is not None:
            return

        self._drone = System()
        print("[DroneEnv] Connecting to PX4 SITL...")
        await self._drone.connect(system_address="udp://:14540")

        async for state in self._drone.core.connection_state():
            if state.is_connected:
                print("[DroneEnv] ✅ Connected to PX4!")
                break

        print("[DroneEnv] Waiting for health OK...")
        async for health in self._drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                print("[DroneEnv] ✅ Health OK")
                break
            print("[DroneEnv]  ...still waiting for health...")
            await asyncio.sleep(1)

        self._connected = True

    async def _arm_and_takeoff(self) -> None:
        """Arm and take off to the target altitude."""
        assert self._drone is not None

        print("[DroneEnv] Arming...")
        await self._drone.action.arm()
        print("[DroneEnv] ✅ Armed")

        print(f"[DroneEnv] Taking off to {self._target_alt} m...")
        await self._drone.action.set_takeoff_altitude(self._target_alt)
        await self._drone.action.takeoff()

        # Give some time to climb
        await asyncio.sleep(8.0)
        print("[DroneEnv] Takeoff complete (approximately).")

    async def _land(self) -> None:
        """Command landing and wait until landed."""
        assert self._drone is not None

        print("[DroneEnv] Landing...")
        await self._drone.action.land()

        async for in_air in self._drone.telemetry.in_air():
            if not in_air:
                print("[DroneEnv] ✅ Landed")
                break
            await asyncio.sleep(1.0)

    async def _get_altitude(self) -> float:
        """Get the current relative altitude (meters)."""
        assert self._drone is not None

        async for pos in self._drone.telemetry.position():
            # relative_altitude_m is altitude above home in meters
            return float(pos.relative_altitude_m)

        # Should never reach here
        return 0.0

    # --------- Gym API: reset / step ---------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and start a new episode."""
        super().reset(seed=seed)
        self._step_count = 0

        async def _reset_coroutine():
            await self._ensure_connection()
            # For now, we always start an episode in the air at target altitude.
            await self._arm_and_takeoff()
            alt = await self._get_altitude()
            return alt

        alt = self._loop.run_until_complete(_reset_coroutine())
        obs = self._build_observation(alt)
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment.

        NOTE: For v0, the action is ignored and we only read the current altitude.
        This is just to get the Gym/PPO loop wired up safely before adding real control.
        """
        self._step_count += 1

        async def _step_coroutine():
            # TODO: In v1, use `action` to send velocity/yaw commands via MAVSDK Offboard.
            alt = await self._get_altitude()
            return alt

        alt = self._loop.run_until_complete(_step_coroutine())

        # Reward: negative absolute error from target altitude
        error = abs(alt - self._target_alt)
        reward = -float(error)

        terminated = False  # No terminal "failure" condition yet
        truncated = self._step_count >= self._max_steps

        obs = self._build_observation(alt)
        info: Dict[str, Any] = {
            "altitude": alt,
            "error": error,
        }

        return obs, reward, terminated, truncated, info

    # --------- Helper: observation builder ---------

    def _build_observation(self, altitude: float) -> np.ndarray:
        """Pack altitude + normalized step into an observation vector."""
        step_fraction = min(1.0, self._step_count / max(1, self._max_steps))
        obs = np.array([altitude, step_fraction], dtype=np.float32)
        return obs


# Simple manual test when running this file directly
if __name__ == "__main__":
    env = DroneEnv(target_altitude=5.0, max_episode_steps=10)
    obs, info = env.reset()
    print("Initial obs:", obs, "info:", info)

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, obs={obs}, info={info}")
        if terminated or truncated:
            break
