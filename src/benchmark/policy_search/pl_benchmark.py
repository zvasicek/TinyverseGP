"""
Benchmark representation module for policy learning benchmarks.
"""

from dataclasses import dataclass

import gymnasium as gym
from src.benchmark.benchmark import Benchmark
from gymnasium.wrappers import FlattenObservation
from ale_py import ALEInterface

from src.gp.tinyverse import Var

@dataclass
class ALEArgs:
    """
    Arguments for the preprocessing of
    Gymnasium Atari Learning Environments (ALE)
    """
    noop_max: int
    frame_skip: int
    screen_size: int
    grayscale_obs: int
    terminal_on_life_loss: int
    scale_obs: int

class PLBenchmark(Benchmark):
    """
    Class for representing policy learning benchmarks as provided
    by Gymnasium. This class currently supports the Box2D, Classic control and ALE environments.
    The ALE environments are ony supported with RGB or grayscale observation space.
    """

    def __init__(self, env_: gym.Env, ale_=False, ale_args: ALEArgs = None):
        self.env = env_
        self.ale = ale_
        self.generate(args=ale_args)

    def generate(self, args: any):
        """
        Wraps the Gym environment either into a flat wrapper or into a
        Atari preprocessing wrapper.

        :param args:
        :return:
        """
        if not self.ale:
            self.wrapped_env = FlattenObservation(self.env)
        else:
            if args is not None:
                self.wrapped_env = gym.wrappers.AtariPreprocessing(self.env,
                                                                   noop_max=args.noop_max,
                                                                   frame_skip=args.frame_skip,
                                                                   screen_size=args.screen_size,
                                                                   grayscale_obs=args.grayscale_obs,
                                                                   terminal_on_life_loss=args.terminal_on_life_loss,
                                                                   scale_obs=args.scale_obs)
            else:
                self.wrapped_env = gym.wrappers.AtariPreprocessing(self.env)
            # self.wrapped_env = gym.wrappers.FrameStack(self.wrapped_env, 4)

    def len_observation_space(self):
        """
        Returns the size of the observation space. The size of an ALE environment
        is of course squared since the input is a frame.
        """
        n = self.wrapped_env.observation_space.shape[0]
        return n if not self.ale else n ** 2

    def len_action_space(self):
        """
        Returns the size of action space.
        """
        return self.env.action_space.n

    def gen_terminals(self):
        """
        Generates and returns the terminals to be used by
        the GP model.
        """
        return [Var(i, 'Inp' + str(i)) for i in range(self.len_observation_space())]