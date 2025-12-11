import os
import gymnasium as gym
import register_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_MODEL_PATH = None      # if None → train from scratch
DEFAULT_TIME_STEPS = 10_000_000
LOGDIR = "snake_logs"


# -----------------------------
# ENV FACTORY
# -----------------------------
def make_env():
    """Factory for monitored Snake environment."""
    def _init():
        env = gym.make("Snake-v0", render_mode="ascii")
        env = Monitor(env)
        return env
    return _init


# -----------------------------
# MAIN TRAINING ENTRY POINT
# -----------------------------
def main(model_path, time_steps):

    # Recreate environment inside main()
    vec_env = DummyVecEnv([make_env()])

    os.makedirs(LOGDIR, exist_ok=True)

    # ---------------------------
    # Load or create model
    # ---------------------------
    if model_path is None:
        print("No model path provided → training from scratch.")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./snake_logs/",
            verbose=1,
        )
        save_prefix = "snake_model"
    else:
        print(f"Loading model from: {model_path}")
        model = PPO.load(model_path, env=vec_env)
        save_prefix = model_path + "_continue"

    # ---------------------------
    # Train
    # ---------------------------
    print("Training starting...")
    model.learn(total_timesteps=time_steps)

    # ---------------------------
    # Save final model
    # ---------------------------
    save_path = save_prefix
    model.save(save_path)
    print(f"Model saved to: {save_path}")


# -----------------------------
# CLI ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to an existing PPO model. Use None to train from scratch.")
    parser.add_argument("--steps", type=int, default=DEFAULT_TIME_STEPS)

    args = parser.parse_args()

    # Convert string "None" from CLI → actual None
    model_path = None if args.model in ["None", "none", "", "null"] else args.model

    main(model_path, args.steps)