import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("Pendulum-v1")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20_000)   # ~2-5 min on a laptop CPU

# victory lap: watch your first trained policy
env = gym.make("Pendulum-v1", render_mode="human")
obs, _ = env.reset()
for _ in range(400):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
env.close()
print("setup complete — see you on day 1")