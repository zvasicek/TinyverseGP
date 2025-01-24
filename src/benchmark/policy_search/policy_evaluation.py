import gymnasium as gym
from src.gp.tinyverse import GPModel
import statistics

class GPAgent:
    def __init__(self, env_: gym.Env):
        self.env = env_

    def evaluate_policy(self, policy, model, num_episodes = 100, wait_key=False):
        rewards = []
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            cumulative_reward = 0
            while not done:
                action = self.get_action(policy, model, obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                obs = next_obs
                cumulative_reward += reward
            if wait_key:
                input("Press Enter to continue...")
            rewards.append(cumulative_reward)
        return statistics.mean(rewards)

    def get_action(self, policy: list[int], model: GPModel, obs):
        prediction = model.predict(policy, obs)
        maximum = max(prediction)
        return prediction.index(maximum)
