"""CLI: demo, train, evaluate, plot, animate, observe; no hidden manuscript targets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .config import Config


def main(argv=None):
    parser=argparse.ArgumentParser(prog='green-cav',description='Synthetic research simulator + optional live observation bridge')
    parser.add_argument('--config',default=None,help='YAML configuration file')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('evaluate',help='Run paired baselines and export CSV')
    run.add_argument('--out',default='outputs/evaluation');run.add_argument('--tasks',type=int,default=None)
    run.add_argument('--seeds',type=int,default=3)
    run.add_argument('--policies',nargs='+',default=['local','edge','greedy','deadline','energy_fixed','proposed_heuristic'])
    run.add_argument('--checkpoint',default=None)
    run.add_argument('--no-task-logs',action='store_true')
    demo=sub.add_parser('demo',help='Smoke test + PNG + MP4 animation')
    demo.add_argument('--out',default='outputs/demo');demo.add_argument('--tasks',type=int,default=350)
    train=sub.add_parser('train',help='Train a new custom PPO policy from scratch')
    train.add_argument('--updates',type=int,default=4);train.add_argument('--checkpoint',default='checkpoints/ppo.pt')
    plot=sub.add_parser('plot',help='Plot an existing experiment directory');plot.add_argument('--out',required=True)
    animate=sub.add_parser('animate',help='Create moving-car animation from task logs')
    animate.add_argument('--csv',required=True);animate.add_argument('--out',default='outputs/traffic.mp4')
    animate.add_argument('--frames',type=int,default=70)
    study=sub.add_parser('study',help='Weather/uplink/density/preference/ablation experiments')
    study.add_argument('--kind',required=True,choices=['weather','uplink','density','preference','ablation'])
    study.add_argument('--tasks',type=int,default=None)
    study.add_argument('--seeds',type=int,default=3)
    study.add_argument('--policy',default='proposed_heuristic')
    study.add_argument('--checkpoint',default=None)
    study.add_argument('--out',default='outputs/studies')
    obs=sub.add_parser('observe',help='Read-only snapshot from externally synchronized CARLA and SUMO')
    obs.add_argument('--out',default='outputs/live_snapshot.json')
    args=parser.parse_args(argv)
    overrides={}
    if getattr(args,'tasks',None) is not None: overrides['n_tasks']=args.tasks
    cfg=Config.load(args.config,**overrides)
    if args.command in ('demo','evaluate'):
        from .experiments import run_suite
        from .plotter import make_plots
        if args.command=='demo':
            policies=['proposed_heuristic','energy_fixed','local','edge']
            seeds=1
        else: policies=args.policies;seeds=args.seeds
        run_suite(cfg,args.out,policies,seeds,getattr(args,'checkpoint',None),not getattr(args,'no_task_logs',False))
        make_plots(args.out)
        if args.command=='demo':
            from .animation import animate_traffic
            task_path=Path(args.out)/f'tasks_proposed_heuristic_seed{cfg.seed}.csv'
            video=Path(args.out)/'moving_vehicles.mp4'
            animate_traffic(cfg,task_path,video)
            print(f'Created: {video}')
    elif args.command=='train':
        from .ppo import train_ppo
        train_ppo(cfg,args.checkpoint,args.updates)
    elif args.command=='plot':
        from .plotter import make_plots
        make_plots(args.out)
    elif args.command=='animate':
        from .animation import animate_traffic
        animate_traffic(cfg,args.csv,args.out,args.frames)
    elif args.command=='study':
        from .experiments import run_study
        run_study(cfg,args.kind,args.out,args.seeds,args.policy,args.checkpoint)
    elif args.command=='observe':
        from .mobility import CarlaSumoObserver
        watcher=CarlaSumoObserver(cfg)
        try: result=watcher.snapshot()
        finally: watcher.close()
        p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('Observed existing co-simulation:',p)

if __name__=='__main__': main()
