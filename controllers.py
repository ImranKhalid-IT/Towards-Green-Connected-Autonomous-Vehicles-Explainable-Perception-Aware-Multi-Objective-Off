"""Explicit baseline heuristics and PPO checkpoint inference.

A heuristic labeled 'proposed_heuristic' is NOT the trained proposed PPO policy.
"""
from __future__ import annotations
import numpy as np

class Controller:
    def act(self,env,obs):
        raise NotImplementedError

class ConstantController(Controller):
    def __init__(self,site,detector=1): self.site=site;self.detector=detector
    def act(self,env,obs): return self.site*4+self.detector

class GreedyLatency(Controller):
    def act(self,env,obs):
        # estimated latency preference; no claimed optimality
        a0=env.alpha
        env.alpha=0
        try: return int(np.argmin([env.predict_action(a) for a in range(8)]))
        finally: env.alpha=a0

class DeadlineAware(Controller):
    def act(self,env,obs):
        # Consider all joint site/model choices: min energy among predicted < deadline.
        values=[]
        alpha=env.alpha
        for a in range(8):
            env.alpha=1.0
            energy_score=env.predict_action(a)
            env.alpha=0.0
            latency_score=env.predict_action(a)
            values.append((a,energy_score,latency_score))
        env.alpha=alpha
        feasible=[v for v in values if v[2] < 1.0]
        return int(min(feasible or values,key=lambda x:x[1] if feasible else x[2])[0])

class ProposedHeuristic(Controller):
    def act(self,env,obs):
        actions=[a for a in range(8) if env.cfg.fixed_detector is None or a%4==env.cfg.fixed_detector]
        return int(min(actions,key=env.predict_action))

class EnergyAwareFixedDetector(Controller):
    def __init__(self,detector=1): self.detector=detector
    def act(self,env,obs):
        return int(min((self.detector,4+self.detector),key=lambda a:env.predict_action(a)))

class RandomController(Controller):
    def __init__(self,seed=0): self.rng=np.random.default_rng(seed)
    def act(self,env,obs):return int(self.rng.integers(0,8))

def controller_from_name(name,seed=0,checkpoint=None):
    if name=='local': return ConstantController(0)
    if name=='edge': return ConstantController(1)
    if name=='greedy':return GreedyLatency()
    if name=='deadline':return DeadlineAware()
    if name=='energy_fixed':return EnergyAwareFixedDetector()
    if name=='proposed_heuristic':return ProposedHeuristic()
    if name=='random':return RandomController(seed)
    if name=='ppo':
        if not checkpoint: raise ValueError('PPO needs --checkpoint path to trained weights')
        from .ppo import PPOController
        return PPOController(checkpoint)
    raise ValueError(f'Unknown controller: {name}')
