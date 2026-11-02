"""Stationary-distribution-preserving Markov radio and measured-equivalent V2I abstraction."""
from __future__ import annotations
import numpy as np

class MarkovChannel:
    def __init__(self,cfg,seed):
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)
        p=np.asarray(cfg.link_stationary, dtype=float)
        self.matrix=cfg.link_persistence*np.eye(3)+(1-cfg.link_persistence)*np.tile(p,(3,1))
        self.state=int(self.rng.choice(3,p=p))
    def next(self) -> int:
        self.state=int(self.rng.choice(3,p=self.matrix[self.state]))
        return self.state
    def uplink_ms(self,s:int) -> float:
        return 8*self.cfg.feature_bytes/(self.cfg.link_rates_mbps[s]*1e6)*1e3+self.cfg.link_overheads_ms[s]
    def modem_power_w(self,s:int) -> float:
        # Calibrated from paper's E_tx Table 8: 0.093,0.045,0.028J / modeled T_up.
        return (0.093,0.045,0.028)[s]/(self.uplink_ms(s)/1000.)
    def energy_j(self,s:int) -> float:
        return self.modem_power_w(s)*self.uplink_ms(s)/1000.
    def successful(self,s:int) -> bool:
        return bool(self.rng.random()<self.cfg.link_success[s])
