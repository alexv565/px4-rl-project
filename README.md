## Current Status

- Docker image with Ubuntu 22.04, PX4 SITL, Gazebo, MAVSDK, and Stable-Baselines3 builds successfully.
- Can run PX4 SITL with `make px4_sitl gz_x500` inside the container.
- `scripts/hover_demo.py` connects via MAVSDK and performs: arm → takeoff → hover → land.
- `rl/envs/drone_env.py` implements a Gymnasium environment on top of PX4 + MAVSDK.
- `rl/train/train_ppo.py` runs a PPO training loop against DroneEnv and saves a test model.

