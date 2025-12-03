from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from rl.envs.drone_env import DroneEnv  # same path you use in train_ppo.py


def make_env(target_altitude: float = 5.0, max_episode_steps: int = 250):
    """
    Environment factory for HPC runs.
    NOTE: This assumes DroneEnv can run headless on HPC. If DroneEnv
    requires PX4 SITL, you will later want a pure-sim env for HPC.
    """
    env = DroneEnv(target_altitude=target_altitude,
                   max_episode_steps=max_episode_steps)
    env = Monitor(env)
    return env


def main():
    # Hyperparameters for HPC-scale training
    TOTAL_TIMESTEPS = int(3e6)
    TARGET_ALTITUDE = 5.0
    MAX_EPISODE_STEPS = 250

    # Create training & eval envs
    train_env = make_env(TARGET_ALTITUDE, MAX_EPISODE_STEPS)
    eval_env = make_env(TARGET_ALTITUDE, MAX_EPISODE_STEPS)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_hpc/best/",
        log_path="./models_hpc/eval_logs/",
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs_hpc/",
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
    )

    model.save("./models_hpc/ppo_drone_final")
    print("[HPC] Training complete.")


if __name__ == "__main__":
    main()
