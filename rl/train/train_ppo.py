from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from rl.envs.drone_env import DroneEnv


def main():
    # --- Training Hyperparameters ---
    TOTAL_TIMESTEPS = 200_000         # Real training run
    EPISODE_STEPS = 250               # Longer episodes for altitude control
    TARGET_ALT = 5.0

    # Create training and evaluation environments
    train_env = Monitor(DroneEnv(target_altitude=TARGET_ALT, max_episode_steps=EPISODE_STEPS))
    eval_env = Monitor(DroneEnv(target_altitude=TARGET_ALT, max_episode_steps=EPISODE_STEPS))

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./models/eval_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="./models/checkpoints/",
        name_prefix="ppo_drone_altitude"
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
    )

    # Train model
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback]
    )

    # Save final model
    model.save("./models/ppo_drone_final")
    print("Full training complete.")


if __name__ == "__main__":
    main()
