import gymnasium as gym
from src.gp.tinyverse import GPModel

class GPAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def evaluate_policy(self, policy, model):
        obs, info = self.env.reset()
        done = False
        cumulative_reward = 0
        while not done:
            action = self.get_action(obs, policy, model)
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            done = terminated or truncated
            obs = next_obs
            cumulative_reward += reward

    def get_action(self, policy: list[int], model: GPModel, obs):
        return model.evaluate_obervation(policy, obs)