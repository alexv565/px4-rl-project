FROM ubuntu:22.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# 1) Basic tools and Python
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    sudo \
    lsb-release \
    gnupg \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*



# 3) Clone PX4 source
RUN git clone https://github.com/PX4/PX4-Autopilot.git /opt/PX4

# 4) Install PX4 dependencies (non-interactive best effort)
WORKDIR /opt/PX4
RUN bash ./Tools/setup/ubuntu.sh --no-nuttx || true

# 5a) Install Python libraries for RL + MAVSDK (excluding torch)
RUN pip3 install --no-cache-dir \
    mavsdk==2.* \
    gymnasium \
    stable-baselines3 \
    numpy \
    pandas \
    tensorboard

# 5b) Install CPU-only PyTorch from the official wheel index
RUN pip3 install --no-cache-dir \
    torch==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu

# 6) Workspace setup
WORKDIR /workspace
ENV PYTHONPATH=/workspace

# 7) Default command: just give us a shell
CMD ["/bin/bash"]