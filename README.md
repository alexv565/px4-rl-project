# PX4 + Gazebo Reinforcement Learning Project

## Environment Versions
- Ubuntu: 22.04 (Docker container)
- PX4: (current branch)
- Gazebo: Harmonic
- Python: 3.x
- PyTorch: 2.3.0 (CPU)
- SB3: latest
- MAVSDK: 2.x

## Container Commands
docker build --no-cache -t px4-rl:latest .
docker run --rm -it px4-rl:latest /bin/bash

## Next Steps
- Step 3: Write MAVSDK takeoff script.
