from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from rl.envs.drone_env import DroneEnv


def main():
    # Create environment
    env = DroneEnv(target_altitude=5.0, max_episode_steps=20)
    env = Monitor(env)

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    # Short test run just to verify integration (not real training yet)
    model.learn(total_timesteps=200)

    # Save model
    model.save("ppo_drone_test")
    print("Training run complete and model saved.")


if __name__ == "__main__":
    main()
