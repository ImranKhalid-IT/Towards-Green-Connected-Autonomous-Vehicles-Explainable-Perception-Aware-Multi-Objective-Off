"""Actual MP4 traffic animation of synthetic motion and measured model outputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,FFMpegWriter,PillowWriter
from .mobility import SyntheticMobility


def animate_traffic(cfg,task_csv,outpath,frames=70,fps=12):
    df=pd.read_csv(task_csv).sort_values('timestamp_s')
    if len(df)<1: raise ValueError('No task telemetry')
    mobile=SyntheticMobility(cfg,int(df.seed.iloc[0])+1)
    fig,ax=plt.subplots(figsize=(11,5.4),dpi=110)
    fig.patch.set_facecolor('#0c1522');ax.set_facecolor('#192636')
    ax.set_xlim(-121,120);ax.set_ylim(-8,17)
    ax.set_xticks([]);ax.set_yticks([])
    for y in [-2,2,6,10]:
        ax.hlines(y,-121,121,color='#71839b',linewidth=0.7,alpha=0.7)
    ax.text(-115,14,'Synthetic V2I perception offloading | CARLA-style schematic',color='white',fontsize=12,weight='bold')
    ax.scatter([0],[12],s=260,c='#f4cc71',marker='s',zorder=5)
    ax.text(5,12,'RSU / MEC',color='#f4cc71',va='center',fontsize=10)
    markers=[]
    for v in mobile.vehicles:
        p=ax.plot([0],[0],marker='s',markersize=11,color='#79d9ed',linestyle='None',zorder=6)[0]
        markers.append(p)
    conn=ax.plot([],[],linewidth=1.2,color='#f4cc71',linestyle='--',alpha=.8)[0]
    status=ax.text(-115,-6.5,'',color='white',fontsize=10)
    stop=min(float(df.timestamp_s.max()),max(0,frames/fps))
    ts=np.linspace(0,stop,max(2,frames))
    times=df.timestamp_s.to_numpy()
    def draw(frame):
        t=ts[frame]
        for idx,v in enumerate(mobile.vehicles):
            x=(v.x0+v.velocity_ms*t)%240-120
            markers[idx].set_data([x],[v.lane*4])
        nearest=int(np.searchsorted(times,t,side='right')-1)
        nearest=max(0,min(len(df)-1,nearest))
        d=df.iloc[nearest]
        if d.offloaded:
            v=mobile.vehicles[int(d.vehicle_id)]
            x=(v.x0+v.velocity_ms*t)%240-120
            conn.set_data([x,0],[v.lane*4,12])
        else: conn.set_data([],[])
        status.set_text(f't={t:5.2f}s  vehicle={int(d.vehicle_id)}  {d.detector}  '
               f'{"EDGE" if d.offloaded else "LOCAL"}   link={d.link}  '
               f'latency={d.latency_ms:.1f} ms   energy={d.system_j:.2f} J   XAI={int(d.xai_invoked)}')
        return (*markers,conn,status)
    anim=FuncAnimation(fig,draw,frames=len(ts),interval=1000/fps,blit=True)
    out=Path(outpath);out.parent.mkdir(parents=True,exist_ok=True)
    if out.suffix.lower()=='.gif': anim.save(str(out),writer=PillowWriter(fps=fps))
    else: anim.save(str(out),writer=FFMpegWriter(fps=fps,codec='libx264',extra_args=['-pix_fmt','yuv420p','-crf','24']))
    plt.close(fig)
    return out
