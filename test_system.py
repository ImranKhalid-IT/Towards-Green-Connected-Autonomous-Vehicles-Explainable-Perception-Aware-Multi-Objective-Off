from dataclasses import replace
import numpy as np
from green_cav.config import Config
from green_cav.channel import MarkovChannel
from green_cav.simulator import GreenCAVEnv
from green_cav.experiments import run_one
from green_cav.analysis import aggregate_run
from green_cav.ppo import compute_gae
from green_cav.models import Action

def test_markov_stationary():
    cfg=Config()
    chan=MarkovChannel(cfg,77)
    dist=np.asarray(cfg.link_stationary)
    assert np.allclose(dist@chan.matrix,dist)
    assert abs(chan.uplink_ms(0)-45.2)<1e-8
    assert abs(chan.uplink_ms(2)-8.9666666667)<1e-5

def test_weather_exact_and_actions():
    cfg=Config(n_tasks=100,n_vehicles=5)
    env=GreenCAVEnv(cfg,seed=10)
    assert np.bincount(env.mobility.weather,minlength=4).tolist()==[87,8,3,2]
    for a in range(8):assert Action.from_id(a).index==a

def test_run_and_energy_consistency():
    cfg=Config(n_tasks=70,seed=7)
    df=run_one(cfg,'proposed_heuristic',7)
    assert len(df)==70
    assert np.isfinite(df[['latency_ms','system_j','cost']]).all().all()
    assert (df.system_j>=df.vehicle_j).all()
    assert (df.deadline_violation*df.dropped==0).all()
    assert (df.latency_ms>0).all()
    metric=aggregate_run(df)
    assert 0<=metric['xai_invocation_pct']<=100
    assert metric['actual_map_50'].startswith('NOT_COMPUTED')

def test_identical_seeds_reproducible():
    cfg=Config(n_tasks=55)
    a=run_one(cfg,'proposed_heuristic',123)
    b=run_one(cfg,'proposed_heuristic',123)
    assert np.allclose(a.latency_ms,b.latency_ms)
    assert np.allclose(a.system_j,b.system_j)

def test_gae_terminal_mask():
    a,r=compute_gae([1.,2.],[.5,.7],[False,True],50.,0.99,0.95)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(r))
    assert abs(a[1]-(2.-.7))<1e-5
