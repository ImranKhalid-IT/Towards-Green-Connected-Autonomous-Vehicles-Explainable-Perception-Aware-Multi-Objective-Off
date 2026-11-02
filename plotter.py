"""Research-style charts derived only from task logs (no copy of manuscript numbers)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plots(result_dir):
    root=Path(result_dir)
    table=pd.read_csv(root/'run_metrics.csv')
    means=table.groupby('policy',sort=False).mean(numeric_only=True)
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    columns=[('latency_mean_ms','Latency (ms)'),('system_energy_j_per_task','System energy (J/task)'),
             ('deadline_violation_pct','Deadline violations (%)'),('accuracy_proxy_pct','Modeled accuracy proxy (%), NOT mAP')]
    for ax,(col,title) in zip(axs.flat,columns):
        ax.bar(np.arange(len(means)),means[col])
        ax.set_xticks(range(len(means)),means.index,rotation=30,ha='right',fontsize=8)
        ax.set_ylabel(title);ax.grid(axis='y',alpha=0.25)
    fig.suptitle('New synthetic simulation outputs (not manuscript Table 3)',fontsize=14)
    fig.tight_layout()
    path=root/'policy_comparison.png';fig.savefig(path,dpi=170);plt.close(fig)
    for csv in sorted(root.glob('tasks_*.csv'))[:1]:
        df=pd.read_csv(csv)
        fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
        axs[0].plot(df.task,df.latency_ms,lw=0.8,label='Task latency')
        axs[0].axhline(100,linestyle='--',color='gray',label='100ms deadline')
        axs[0].set_ylabel('Latency (ms)');axs[0].legend()
        axs[1].plot(df.task,df.system_j.cumsum(),lw=1,label='System cumulative energy')
        axs[1].set_ylabel('Energy (J)');axs[1].set_xlabel('Perception task index')
        axs[1].legend();fig.tight_layout()
        fig.savefig(root/'time_series.png',dpi=170);plt.close(fig)
    return path
