"""Multi-server future completion ledger + scheduled Poisson background arrivals."""
from __future__ import annotations
import heapq
import numpy as np

class ServiceQueue:
    """Deterministic multi-server FCFS reservations with future completion records.

    Arrival requests MUST be ordered by time. Every reserved job is retained in a
    completion heap, including jobs waiting for a server, to enforce buffer limits.
    """
    def __init__(self, streams:int, max_buffer:int):
        self.streams=streams
        self.max_buffer=max_buffer
        self.completions=[]
        self.server_ready=[0.0]*streams
        self.last_time=-float('inf')
    def _prune(self,t):
        if t < self.last_time-1e-9:
            raise ValueError('Out-of-order queue arrival; advance network events in time order')
        self.last_time=t
        while self.completions and self.completions[0]<=t+1e-9:
            heapq.heappop(self.completions)
    def status(self,t):
        outstanding=sum(1 for e in self.completions if e>t)
        return max(0,outstanding-self.streams)
    def waiting_ms(self,t):
        return max(0.,(min(self.server_ready)-t)*1000.)
    def enqueue(self,t:float, service_ms:float):
        self._prune(t)
        depth=max(0,len(self.completions)-self.streams)
        if len(self.completions)>=self.streams+self.max_buffer:
            return None, depth
        stream=int(np.argmin(self.server_ready))
        start=max(t,self.server_ready[stream])
        wait_ms=max(0.,(start-t)*1000.)
        completion=start+service_ms/1000.
        self.server_ready[stream]=completion
        heapq.heappush(self.completions,completion)
        return wait_ms,depth

class QueueFabric:
    def __init__(self,cfg,seed):
        self.cfg=cfg
        self.rng=np.random.default_rng(seed)
        self.edge=ServiceQueue(cfg.edge_workers,cfg.edge_max_buffer)
        self.local=[ServiceQueue(1,cfg.local_max_buffer) for _ in range(cfg.n_vehicles)]
        self.next_background=(float(self.rng.exponential(1/cfg.background_edge_rate_hz))
                              if cfg.background_edge_rate_hz>0 else float('inf'))
        self.bg_service_ms=20.0
        self.pending_bg=[]
        self.edge_load_ewma=0.35
    def reserve_local(self,vehicle,t,service_ms):
        return self.local[vehicle].enqueue(t,service_ms)
    def reserve_edge(self,t,service_ms):
        while self.next_background<=t:
            arrival=self.next_background
            bg_service=float(self.bg_service_ms*self.rng.lognormal(0,0.15))
            self.edge.enqueue(arrival,bg_service)
            self.next_background+=float(self.rng.exponential(1/self.cfg.background_edge_rate_hz))
        return self.edge.enqueue(t,service_ms)
    def edge_depth(self,t):
        return self.edge.status(t)
    def local_depth(self,vehicle,t):
        return self.local[vehicle].status(t)
    def estimated_load(self,arrivals_per_s):
        return min(1.5,(arrivals_per_s+self.cfg.background_edge_rate_hz)*self.bg_service_ms/1000/self.cfg.edge_workers)
