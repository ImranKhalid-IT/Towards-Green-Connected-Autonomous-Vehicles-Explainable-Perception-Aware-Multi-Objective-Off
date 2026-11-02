"""Gated uncertainty feedback, synthetic alignment, optional trainable XAI head.

DualHeadXAI is an untrained architecture until supervised optimization on real
feature tensors / object masks; surrogate alignment is clearly marked synthetic.
"""
from __future__ import annotations
import numpy as np

class UncertaintyGate:
    def __init__(self,cfg,seed):
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)
        self.prev_u=0.15
        self.prev_r=0.78
        self.threshold=None
        self.calibrate()
    def calibrate(self):
        # The surrogate's constructed class probabilities have the chosen entropy.
        # Vectorized validation calibration avoids repeated neural-model inference.
        rng=np.random.default_rng(177)
        weather=rng.choice(4,15000,p=self.cfg.weather_probs)
        detector=rng.integers(4,size=15000)
        bases=np.array((0.12,0.26,0.43,0.62))
        us=np.clip(bases[weather]+(2-detector)*0.026+rng.normal(0,0.065,15000),0.015,0.97)
        self.threshold=float(np.quantile(us,1-self.cfg.xai_budget))
        return self.threshold
    def process(self,uncertainty,weather,detector,enabled=True):
        triggered=bool(enabled and uncertainty>=self.threshold)
        if triggered:
            # Not a real saliency map: surrogate object-alignment score.
            reliability=float(np.clip(0.87 - 0.38*uncertainty - 0.028*weather
                          +0.012*detector+self.rng.normal(0,0.045),0,1))
            fused=(1-self.cfg.fusion_weight)*uncertainty +self.cfg.fusion_weight*(1-reliability)
        else:
            reliability=self.prev_r
            fused=uncertainty
        self.prev_u=float(fused)
        self.prev_r=float(reliability)
        return float(fused),float(reliability),int(triggered)

try:
    import torch
    from torch import nn
    class DualHeadXAI(nn.Module):
        """Trainable feature-to-saliency + confidence/reliability heads.

        Input [B,C,H,W], saliency [B,1,H,W], reliability [B,1].
        No training dataset/checkpoints are supplied with the manuscript.
        """
        def __init__(self,feature_channels=64,width=32):
            super().__init__()
            self.trunk=nn.Sequential(nn.Conv2d(feature_channels,width,3,padding=1),
                       nn.BatchNorm2d(width),nn.SiLU(),nn.Conv2d(width,width,3,padding=1),nn.SiLU())
            self.saliency_head=nn.Conv2d(width,1,1)
            self.reliability_head=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),
                                               nn.Linear(width,16),nn.SiLU(),nn.Linear(16,1))
        def forward(self,features):
            z=self.trunk(features)
            return torch.sigmoid(self.saliency_head(z)),torch.sigmoid(self.reliability_head(z))

    def xai_supervised_step(model,optimizer,features,object_masks,alignment_targets):
        """One supervised update; targets must be dataset-grounded rather than synthetic."""
        import torch.nn.functional as F
        model.train()
        sal,rel=model(features)
        loss=F.binary_cross_entropy(sal,object_masks)+0.5*F.mse_loss(rel,alignment_targets)
        optimizer.zero_grad();loss.backward();optimizer.step()
        return float(loss.item())
except ImportError:
    DualHeadXAI=None
    def xai_supervised_step(*args,**kwargs):
        raise RuntimeError('PyTorch required for actual dual-head training')
