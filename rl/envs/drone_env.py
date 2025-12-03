import asyncio
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed


class DroneEnv(gym.Env):
    """
    PX4 + MAVSDK Gym environment (v1) with simple vertical Offboard control.

    - Action: 1D continuous value in [-1, 1], mapped to vertical speed (m/s).
      * -1 -> max upward speed
      * +1 -> max downward speed

    - Observation: [altitude_m, normalized_step]
    - Reward: negative absolute error from target altitude, minus small action penalty.
    """

    metadata = {"render_modes": []}

    def __init__(self, target_altitude: float = 5.0, max_episode_steps: int = 200):
        super().__init__()

        # ---- RL interface definitions ----
        # Single continuous action in [-1, 1] controlling vertical speed
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation: [altitude, normalized_step]
        self.observation_space = spaces.Box(
            low=np.array([-1000.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

        # ---- PX4 / MAVSDK related ----
        self._drone: Optional[System] = None
        self._connected: bool = False

        # Single asyncio event loop for this env
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # ---- Episode bookkeeping ----
        self._target_alt = float(target_altitude)
        self._max_steps = int(max_episode_steps)
        self._step_count = 0

        # Control parameters
        self._max_vz = 1.0  # m/s, maximum vertical speed magnitude
        self._dt = 0.3      # seconds per step
        self._last_alt = 0.0

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
            await asyncio.sleep(1.0)

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

    async def _start_offboard(self) -> None:
        """Start Offboard mode with zero initial velocity setpoint."""
        assert self._drone is not None

        print("[DroneEnv] Starting Offboard mode...")
        # Offboard requires a setpoint before starting
        await self._drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        try:
            await self._drone.offboard.start()
            print("[DroneEnv] ✅ Offboard started")
        except OffboardError as e:
            print(f"[DroneEnv] Offboard start failed: {e._result.result}")
            # As a simple recovery, just land
            await self._drone.action.land()
            raise

    async def _stop_offboard_and_land(self) -> None:
        """Stop Offboard safely and land."""
        assert self._drone is not None

        print("[DroneEnv] Stopping Offboard and landing...")
        try:
            await self._drone.offboard.stop()
        except OffboardError as e:
            print(f"[DroneEnv] Offboard stop failed: {e._result.result}")

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
            return float(pos.relative_altitude_m)

        return 0.0

    async def _apply_action_and_wait(self, action: np.ndarray) -> float:
        """
        Apply the given action as a vertical velocity command and wait dt seconds,
        then return the new altitude.
        """
        assert self._drone is not None

        # Clip and map action to vertical speed
        a = float(np.clip(action[0], -1.0, 1.0))
        # PX4 NED convention: vz > 0 is down, vz < 0 is up
        vz_cmd = -a * self._max_vz  # so a=+1 => go down, a=-1 => go up

        await self._drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_m_s=0.0,
                right_m_s=0.0,
                down_m_s=vz_cmd,
                yawspeed_deg_s=0.0,
            )
        )

        await asyncio.sleep(self._dt)
        alt = await self._get_altitude()
        return alt

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
            await self._arm_and_takeoff()
            await self._start_offboard()
            alt = await self._get_altitude()
            return alt

        alt = self._loop.run_until_complete(_reset_coroutine())
        self._last_alt = alt
        obs = self._build_observation(alt)
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment:

        - Applies the action as a vertical velocity command via Offboard.
        - Waits dt seconds.
        - Reads the new altitude.
        - Computes reward based on altitude error and action magnitude.
        """
        self._step_count += 1

        async def _step_coroutine():
            alt = await self._apply_action_and_wait(action)
            return alt

        alt = self._loop.run_until_complete(_step_coroutine())

        # Compute reward: negative absolute altitude error + small action penalty
        error = abs(alt - self._target_alt)
        action_mag = float(abs(np.clip(action[0], -1.0, 1.0)))
        reward = -float(error) - 0.1 * action_mag

        # Termination conditions
        terminated = False
        # Consider it "done" if we go way out of safe bounds
        if alt < 0.5 or alt > 15.0:
            terminated = True

        truncated = self._step_count >= self._max_steps

        obs = self._build_observation(alt)
        info: Dict[str, Any] = {
            "altitude": alt,
            "error": error,
            "action_mag": action_mag,
        }

        # If episode ended, try to stop offboard and land
        if terminated or truncated:
            async def _end_coroutine():
                if self._drone is not None:
                    await self._stop_offboard_and_land()
            self._loop.run_until_complete(_end_coroutine())

        self._last_alt = alt
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

    for i in range(10):
        # Random vertical speed commands
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, obs={obs}, info={info}")
        if terminated or truncated:
            break
