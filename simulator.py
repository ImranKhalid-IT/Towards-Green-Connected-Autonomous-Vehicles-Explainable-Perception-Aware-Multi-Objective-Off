"""Per-task MDP with shared modeled channel, queues, perception and energy accounting.

Synthetic environment only. Queue service reservation at task RELEASE is an
approximation to a true arrival-ordered edge-network event scheduler; it prevents
future uplink completions from being processed out of time order during online RL.
"""
from __future__ import annotations
import numpy as np
from .models import Action, Outcome, WEATHERS, LINKS, DETECTORS, validate_state
from .mobility import SyntheticMobility
from .channel import MarkovChannel
from .queueing import QueueFabric
from .detectors import PROFILES, SyntheticDetectorBank
from .explainability import UncertaintyGate

class GreenCAVEnv:
    def __init__(self,cfg,seed=None):
        self.cfg=cfg
        self.seed=int(cfg.seed if seed is None else seed)
        self.reset(self.seed)
    def reset(self,seed=None,alpha=None):
        self.seed=int(self.seed if seed is None else seed)
        self.rng=np.random.default_rng(self.seed+73)
        self.mobility=SyntheticMobility(self.cfg,self.seed+1)
        self.channels=[MarkovChannel(self.cfg,self.seed+10+i) for i in range(self.cfg.n_vehicles)]
        self.queues=QueueFabric(self.cfg,self.seed+20)
        self.detectors=SyntheticDetectorBank(self.seed+30)
        self.xai=[UncertaintyGate(self.cfg,self.seed+40+v) for v in range(self.cfg.n_vehicles)]
        self.alpha=float(self.cfg.alpha if alpha is None else alpha)
        self.i=0
        self.task=self._next_task()
        return self.observe()
    def _next_task(self):
        link=self.channels[self.i%self.cfg.n_vehicles].next()
        return self.mobility.task(self.i,link)
    def observe(self):
        task=self.task
        weather_p=np.full(4,0.02,dtype=np.float32)
        weather_p[task.weather]=0.94  # synthetic weather estimator, not inference from pixels
        t=task.timestamp_s
        link_mbps=self.cfg.link_rates_mbps[task.link]
        # approx forecast based on running backlog and background load.
        n_edge=self.queues.edge_depth(t)
        queue_ms=(n_edge*20/self.cfg.edge_workers)
        load=self.queues.estimated_load(0.70*self.cfg.n_vehicles*self.cfg.sensing_hz)
        state=np.array([*weather_p,
            self.xai[task.vehicle_id].prev_u if self.cfg.feedback_enabled else 0.15,
            self.xai[task.vehicle_id].prev_r if self.cfg.feedback_enabled else 0.78,
            link_mbps/30., queue_ms/100.,min(load,1.5),
            min(1.,self.queues.local_depth(task.vehicle_id,t)/max(1,self.cfg.local_max_buffer)),
            self.alpha,1-self.alpha],dtype=np.float32)
        return validate_state(state)
    def predict_action(self,action_id):
        a=Action.from_id(action_id)
        task=self.task
        p=PROFILES[a.detector]
        up=self.channels[task.vehicle_id].uplink_ms(task.link) if a.site else 0
        service=p.edge_ms if a.site else p.local_ms
        edge_wait=max(0,self.queues.edge_depth(task.timestamp_s))*20/max(1,self.cfg.edge_workers) if a.site else 0
        approx_xai=(self.cfg.xai_budget*self.cfg.xai_call_ms if self.cfg.enable_xai else 0.0)
        latency=self.cfg.pre_ms+self.cfg.decision_ms+approx_xai+service+up+edge_wait+(self.cfg.return_ms if a.site else 0)
        vehicle=(self.cfg.preprocess_energy_j +
                 (self.channels[task.vehicle_id].energy_j(task.link) if a.site else p.local_j)+
                 self.cfg.local_static_w*latency/1000)
        energy=vehicle+(p.edge_j if a.site else 0)+(self.cfg.xai_budget if self.cfg.enable_xai else 0)*(self.cfg.xai_edge_energy_j if a.site else self.cfg.xai_loc_energy_j)
        accuracy=self.detectors.expected_accuracy(a.detector,task.weather)/100.
        deadline_risk=max(0,latency-self.cfg.deadline_ms)/self.cfg.deadline_ms
        cost=(self.alpha*energy/self.cfg.energy_ref_j+(1-self.alpha)*latency/self.cfg.deadline_ms +
              self.cfg.accuracy_weight*(1-accuracy)+self.cfg.deadline_penalty*deadline_risk)
        return float(cost)
    def step(self,action_id):
        a=Action.from_id(action_id)
        task=self.task
        cfg=self.cfg
        ch=self.channels[task.vehicle_id]
        compute_ms,compute_j=self.detectors.cost_profile(a.detector,a.site)
        # Independent service-time jitter; profiles represent single-inference means.
        compute_ms*=float(self.rng.lognormal(-cfg.service_jitter_sigma**2/2,cfg.service_jitter_sigma))
        acc,unc=self.detectors.predict(a.detector,task.weather)
        fused,reliability,xai=self.xai[task.vehicle_id].process(unc,task.weather,a.detector,enabled=cfg.enable_xai)
        up_ms=ch.uplink_ms(task.link) if a.site else 0.0
        # Queue reservation uses task RELEASE time, not true post-uplink arrival time.
        if a.site:
            q_ms,depth=self.queues.reserve_edge(task.timestamp_s,compute_ms)
            link_success=ch.successful(task.link)
            dropped=(not link_success) or (q_ms is None)
        else:
            q_ms,depth=self.queues.reserve_local(task.vehicle_id,task.timestamp_s,compute_ms)
            dropped=q_ms is None
        q_ms=float(q_ms or 0)
        latency=cfg.pre_ms+cfg.decision_ms+(cfg.return_ms if a.site else 0)+up_ms+q_ms+compute_ms+xai*cfg.xai_call_ms
        # Tasks that drop still incur incurred/stage-estimated workload cost; this is a modeling choice.
        vehicle_j=(cfg.preprocess_energy_j+
                   (ch.energy_j(task.link) if a.site else compute_j)+
                   (xai*cfg.xai_loc_energy_j if not a.site else 0)+
                   cfg.local_static_w*latency/1000)
        system_j=vehicle_j+(compute_j+xai*cfg.xai_edge_energy_j if a.site else 0)
        # Paper counts dropped tasks separately from late completed tasks.
        v=int((not dropped) and latency>cfg.deadline_ms)
        q=(1-int(dropped))*acc/100
        cost=(self.alpha*system_j/cfg.energy_ref_j+
              (1-self.alpha)*latency/cfg.deadline_ms+
              cfg.accuracy_weight*(1-q)+cfg.deadline_penalty*v+cfg.drop_penalty*int(dropped))
        out=Outcome(task.id,self.seed,task.vehicle_id,task.timestamp_s,
             WEATHERS[task.weather],LINKS[task.link],DETECTORS[a.detector],a.site,
             task.x,task.y,task.density,cfg.link_rates_mbps[task.link],q_ms,depth,
             latency,vehicle_j,system_j,acc,fused,reliability,xai,int(dropped),v,cost,-cost)
        self.i+=1
        done=self.i>=cfg.n_tasks
        if not done:
            self.task=self._next_task()
            nxt=self.observe()
        else:
            nxt=np.zeros(12,dtype=np.float32)
        return nxt,float(-cost),done,out
