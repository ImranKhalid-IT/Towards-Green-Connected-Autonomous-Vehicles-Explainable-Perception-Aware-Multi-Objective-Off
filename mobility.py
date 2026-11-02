"""Reproducible synthetic mobility plus optional read-only CARLA+SUMO observer.

This observer DOES NOT synchronize the simulators, produce real object detections,
or alter sensor frames. CARLA's official run_synchronization.py must run separately.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .models import Task, WEATHERS

@dataclass
class Vehicle:
    id: int
    lane: int
    x0: float
    velocity_ms: float

class SyntheticMobility:
    def __init__(self, cfg, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.vehicles = [Vehicle(i, i % 3, float(self.rng.uniform(-110, 10)),
                                 float(self.rng.uniform(8, 20))) for i in range(cfg.n_vehicles)]
        probs = np.array(cfg.weather_probs, dtype=float)
        if cfg.scenario != 'mixed':
            self.weather = np.full(cfg.n_tasks, WEATHERS.index(cfg.scenario), dtype=int)
        elif cfg.exact_weather_shares:
            counts = np.floor(cfg.n_tasks*probs).astype(int)
            for i in np.argsort(-(cfg.n_tasks*probs-counts))[:cfg.n_tasks - sum(counts)]:
                counts[i] += 1
            self.weather = np.repeat(np.arange(4), counts)
            self.rng.shuffle(self.weather)
        else:
            self.weather = self.rng.choice(4, cfg.n_tasks, p=probs)

    def task(self, i: int, link: int) -> Task:
        v = self.vehicles[i % len(self.vehicles)]
        # Each vehicle emits one task each 100 ms. Interleave vehicle releases.
        t = (i // len(self.vehicles)) / self.cfg.sensing_hz + (i % len(self.vehicles)) / (self.cfg.sensing_hz*len(self.vehicles))
        x = (v.x0 + v.velocity_ms*t) % 240 - 120
        density = float(np.clip(0.35 + 0.21*np.sin(t*0.12) + 0.17*np.sin(t*0.051+0.6),0.04,0.95))
        return Task(i,t,v.id,int(self.weather[i]),link,density,x, v.lane*4.0)

class CarlaSumoObserver:
    """Observation-only bridge to independently running official CARLA-SUMO co-sim.

    Expects CARLA 0.9.15 server and a *running* SUMO TraCI server on configured port.
    Read vehicle positions and weather, not detection ground truth, camera imagery,
    or workload profiles. In absence of simulator, raises clear RuntimeError.
    """
    def __init__(self, cfg):
        try:
            import carla
            import traci
        except ImportError as exc:
            raise RuntimeError('live-observe requires matching CARLA 0.9.15 PythonAPI and SUMO traci') from exc
        self.carla = carla
        self.traci = traci
        self.client = carla.Client(cfg.carla_host, cfg.carla_port)
        self.client.set_timeout(8.0)
        self.world = self.client.get_world()
        try:
            traci.init(cfg.traci_port, host=cfg.carla_host, label='greencav_readonly')
            self.conn = traci.getConnection('greencav_readonly')
        except Exception as exc:
            raise RuntimeError('Start the official CARLA-SUMO synchronization and expose a TraCI server before live-observe') from exc

    def snapshot(self):
        snapshot = self.world.get_snapshot()
        w = self.world.get_weather()
        vehicles = []
        for a in self.world.get_actors().filter('vehicle.*'):
            loc = a.get_location()
            vehicles.append({'id': a.id, 'x': loc.x, 'y': loc.y, 'z': loc.z})
        return {'carla_frame': snapshot.frame,
                'carla_time_s': snapshot.timestamp.elapsed_seconds,
                'rain': float(w.precipitation), 'fog': float(w.fog_density),
                'carla_vehicles': vehicles,
                'sumo_vehicles': len(self.conn.vehicle.getIDList())}

    def close(self):
        try: self.conn.close()
        except Exception: pass
