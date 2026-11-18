"""Figures in the style of the paper (Figs. 6-10) from the CSV files written by the pipeline."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GREY, BLUE, ORANGE, GREEN = "#8a8f98", "#2f6db3", "#d9822b", "#2e8b57"


def _colors(names):
    return [GREEN if n == "Proposed" else GREY for n in names]


def make_figures(out: str) -> None:
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    runs = pd.read_csv(os.path.join(out, "runs_policies.csv"))
    order = list(dict.fromkeys(runs.policy))
    g = runs.groupby("policy").agg(["mean", "std"]).reindex(order)

    # Fig. 6: latency and energy
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(order))
    ax[0].bar(x, g[("latency_ms", "mean")], yerr=g[("latency_ms", "std")], color=_colors(order))
    ax[0].scatter(x, g[("p95_ms", "mean")], marker="D", color="k", zorder=3, label="P95")
    ax[0].axhline(100, ls="--", color="r", lw=1, label="Deadline")
    ax[0].set_ylabel("Latency (ms)"); ax[0].legend(frameon=False)
    ax[1].bar(x, g[("energy_kJ", "mean")], yerr=g[("energy_kJ", "std")], color=_colors(order))
    ax[1].set_ylabel("Energy per run (kJ)")
    for a in ax:
        a.set_xticks(x, order, rotation=30, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_policy_latency_energy.pdf")); plt.close(fig)

    # Fig. 7: reliability and offloading
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, col, lab in zip(ax, ["viol_pct", "drop_pct", "offload_pct"], ["Deadline violations (%)", "Task drops (%)", "Offloaded (%)"]):
        a.barh(order, g[(col, "mean")], color=_colors(order)); a.set_xlabel(lab); a.invert_yaxis()
        a.tick_params(axis="y", labelsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_reliability_offloading.pdf")); plt.close(fig)

    # Fig. 8: Pareto view
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.axvspan(100, max(110, g[("latency_ms", "mean")].max() * 1.05), color="r", alpha=0.08)
    for n in order:
        ax.scatter(g.loc[n, ("latency_ms", "mean")], g.loc[n, ("J_task", "mean")], s=60,
                   color=GREEN if n == "Proposed" else BLUE, marker="*" if n == "Proposed" else "o")
        ax.annotate(n, (g.loc[n, ("latency_ms", "mean")], g.loc[n, ("J_task", "mean")]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean end-to-end latency (ms)"); ax.set_ylabel("Energy per task (J)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_pareto.pdf")); plt.close(fig)

    # Fig. 9: system energy and carbon
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    veh = g[("veh_Wh_h", "mean")]; sysw = g[("Wh_h", "mean")]
    ax[0].bar(x, veh, color=BLUE, label="Vehicle side"); ax[0].bar(x, sysw - veh, bottom=veh, color=ORANGE, label="Edge + network")
    ax[0].set_ylabel("Wh per driving hour"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].bar(x, g[("range_loss_km", "mean")], color=_colors(order)); ax[1].set_ylabel("Range loss (km)")
    ax[2].bar(x, g[("co2_g_km", "mean")], color=_colors(order)); ax[2].set_ylabel("g CO$_2$e/km")
    for a in ax:
        a.set_xticks(x, order, rotation=30, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_energy_carbon.pdf")); plt.close(fig)

    # Fig. 10: weather-conditioned cost of the proposed policy
    wpath = os.path.join(out, "runs_policies_weather.csv")
    if os.path.exists(wpath):
        w = pd.read_csv(wpath)
        w = w[w.policy == "Proposed"].groupby("weather", sort=False).mean(numeric_only=True)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        ax[0].bar(w.index, w.latency_ms, color=BLUE); ax0 = ax[0].twinx()
        ax0.plot(w.index, w.viol_pct, "o-", color="r"); ax[0].set_ylabel("Latency (ms)"); ax0.set_ylabel("Violations (%)")
        ax[1].bar(w.index, w.J_task, color=GREEN); ax[1].set_ylabel("Energy per task (J)")
        for a in ax:
            a.tick_params(axis="x", labelsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_weather_cost.pdf")); plt.close(fig)

    # training curves
    mdir = os.path.join(out, "models")
    logs = [f for f in os.listdir(mdir) if f.endswith("_trainlog.csv")] if os.path.isdir(mdir) else []
    if logs:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for f in sorted(logs):
            d = pd.read_csv(os.path.join(mdir, f))
            ax.plot(d.steps, d.mean_cost.rolling(5, min_periods=1).mean(), label=f.replace("_trainlog.csv", ""))
        ax.set_xlabel("Environment steps"); ax.set_ylabel("Mean per-task cost"); ax.legend(frameon=False, fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "fig_training.pdf")); plt.close(fig)
