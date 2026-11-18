"""Unit tests (run with ``python -m pytest tests`` or ``python tests/test_greencav.py``)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from greencav.channel import Channel, stationary, total_noise_dbm, transition_matrix  # noqa: E402
from greencav.config import SimConfig  # noqa: E402
from greencav.detectors import DETECTORS, DETECTOR_NAMES  # noqa: E402
from greencav.env import GreenCAVEnv, N_ACTIONS, encode_action  # noqa: E402
from greencav.paper_checks import main as paper_checks  # noqa: E402
from greencav.ppo import MLP, PPOAgent, masked_log_softmax  # noqa: E402
from greencav.weather import weather_sequence  # noqa: E402
from greencav import xai  # noqa: E402


def test_table3_profiles():
    assert DETECTOR_NAMES == ["YOLOv5s", "YOLOv8s", "YOLOv8m", "Faster R-CNN"]
    assert abs(DETECTORS["YOLOv8s"].local_s - 0.0528) < 1e-12
    assert abs(DETECTORS["Faster R-CNN"].weather_loss(3) - 21.1) < 1e-9
    assert abs(DETECTORS["YOLOv5s"].weather_loss(3) - 24.4) < 1e-9


def test_noise_and_markov():
    assert abs(total_noise_dbm(10e6, 9) + 95) < 1e-9
    P = transition_matrix([0.05, 0.35, 0.60], 0.2)
    assert np.allclose(stationary(P), [0.05, 0.35, 0.60])


def test_link_shares_empirical():
    cfg = SimConfig()
    ch = Channel(cfg, np.random.default_rng(0))
    states = np.array([ch.step() for _ in range(200_000)])
    assert np.allclose(np.bincount(states, minlength=3) / len(states), [0.05, 0.35, 0.60], atol=0.02)


def test_exact_weather_counts():
    cfg = SimConfig()
    w = weather_sequence(cfg, 30_000, np.random.default_rng(1))
    assert list(np.bincount(w, minlength=4)) == [26_100, 2_400, 900, 600]


def test_xai_equations():
    assert abs(xai.predictive_uncertainty([np.ones(10) / 10]) - 1.0) < 1e-9
    assert xai.predictive_uncertainty([]) == 1.0
    s = np.zeros((10, 10)); s[2:4, 2:4] = 1.0
    assert abs(xai.alignment_reliability(s, [(2, 2, 4, 4)]) - 1.0) < 1e-6
    assert xai.gate(0.5, 0.4) == 1 and xai.gate(0.3, 0.4) == 0
    u = np.random.default_rng(0).random(10_000)
    tau = xai.calibrate_threshold(u, 0.25)
    assert abs((u >= tau).mean() - 0.25) < 0.01
    assert abs(xai.amortised_cost(np.r_[np.ones(25), np.zeros(75)], 15.8)[0] - 3.95) < 1e-9
    img = np.ones((8, 8)); sal = np.random.default_rng(0).random((8, 8)) ** 4
    model = lambda x: float((x * sal).sum() / sal.sum()) + 1e-3   # relies on salient pixels
    d, i = xai.deletion_insertion_auc(model, img, sal)
    assert d < i


def test_env_runs_and_gate_budget():
    cfg = SimConfig()
    env = GreenCAVEnv(cfg, seed=3, n_tasks=4000)
    done, xs = False, []
    while not done:
        _, r, done, out = env.step(encode_action(1, 1))
        xs.append(out.x)
        assert np.isfinite(r)
    assert 0.18 < np.mean(xs) < 0.32          # rho_x = 0.25 for the calibration detector


def test_paired_exogenous_processes():
    cfg = SimConfig()
    e1, e2 = GreenCAVEnv(cfg, seed=5, n_tasks=500), GreenCAVEnv(cfg, seed=5, n_tasks=500)
    s1, s2 = [], []
    for k in range(499):
        s1.append(e1.step(encode_action(0, 0))[3].s)
        s2.append(e2.step(encode_action(1, 3))[3].s)
    assert s1 == s2 and np.array_equal(e1.w, e2.w)


def test_mlp_gradient():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 3], rng)
    x = rng.normal(size=(4, 5))
    g_out = rng.normal(size=(4, 3))
    net.forward(x, cache=True)
    grads = net.backward(g_out)
    W = net.W[0]
    eps, (i, j) = 1e-6, (1, 2)
    W[i, j] += eps; fp = (net.forward(x) * g_out).sum()
    W[i, j] -= 2 * eps; fm = (net.forward(x) * g_out).sum()
    W[i, j] += eps
    assert abs((fp - fm) / (2 * eps) - grads[0][i, j]) < 1e-5


def test_masked_softmax_and_agent():
    mask = np.zeros(N_ACTIONS, bool); mask[[1, 5]] = True
    lp = masked_log_softmax(np.zeros((2, N_ACTIONS)), mask)
    assert np.allclose(np.exp(lp)[:, mask].sum(1), 1.0)
    ag = PPOAgent(SimConfig(hidden=[16, 16]), mask)
    for _ in range(20):
        assert ag.act(np.zeros(13))[0] in (1, 5)


def test_paper_tables():
    assert paper_checks()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("passed", name)
