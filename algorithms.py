"""All algorithms of the paper in one place.

Algorithm 1   uncertainty-gated dual-head XAI feedback     -> greencav.xai.algorithm1_gated_xai
Algorithm 2   preference-conditioned PPO training          -> greencav.ppo.algorithm2_train
Algorithm S1  link-aware split-inference task execution    -> greencav.env.GreenCAVEnv.step
Algorithm S2  paired evaluation and statistical testing    -> greencav.evaluate.algorithmS2_evaluate
"""
from .env import GreenCAVEnv
from .evaluate import algorithmS2_evaluate
from .ppo import algorithm2_train
from .xai import algorithm1_gated_xai


def algorithmS1_execute_task(env: GreenCAVEnv, action: int):
    """Algorithm S1 for the current task of ``env``: returns (next_obs, reward, done, TaskOutcome)."""
    return env.step(action)


__all__ = ["algorithm1_gated_xai", "algorithm2_train", "algorithmS1_execute_task", "algorithmS2_evaluate"]
