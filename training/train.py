from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODEL_DIR = PROJECT_ROOT / "results" / "models"

from agents.dqn.dqn_agent import DqnAgent, DqnConfig
from envs.slice_gym_env import SLICE_NAMES, SliceGymEnv
from envs.metrics import compute_sla_rates, nan_or_round
from monitor import TrainingMonitor


_PRB_TOTAL = 51


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    """Return the most advanced DQN checkpoint by total_steps count."""
    checkpoints = sorted(
        model_dir.glob("dqn_ep*.pt"),
        key=lambda p: int(p.stem.replace("dqn_ep", ""))
    )
    final = model_dir / "dqn_final.pt"

    if final.exists():
        ckpt = torch.load(final, map_location="cpu")
        final_steps = ckpt.get("total_steps", -1)
        if checkpoints:
            ckpt_steps = torch.load(
                checkpoints[-1], map_location="cpu"
            ).get("total_steps", -1)
            return final if final_steps >= ckpt_steps else checkpoints[-1]
        return final

    return checkpoints[-1] if checkpoints else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    type=str,  default="dqn", choices=["dqn"])
    parser.add_argument("--port",     type=int,  default=5555)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--episodes", type=int,  default=None)
    parser.add_argument("--config",   type=str,  default="configs/config.yaml")
    parser.add_argument("--resume",   action="store_true",
                        help="Load latest checkpoint and continue from that episode")
    parser.add_argument("--device",   type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    if args.episodes is None:
        args.episodes = int(cfg["train"]["episodes"])

    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"[train.py] Using device: {device}")
    if device.type == "cuda":
        print(f"[train.py] GPU: {torch.cuda.get_device_name(0)}")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)

    dqn_cfg = DqnConfig(
        obs_dim         = 18,
        action_dim      = 27,
        lr              = float(cfg["dqn"]["lr"]),
        gamma           = float(cfg["dqn"]["gamma"]),
        batch_size      = int(  cfg["dqn"]["batch_size"]),
        buffer_size     = int(  cfg["dqn"]["buffer_size"]),
        tau             = float(cfg["dqn"]["tau"]),
        eps_start       = float(cfg["dqn"]["eps_start"]),
        eps_end         = float(cfg["dqn"]["eps_end"]),
        eps_decay_steps = int(  cfg["dqn"]["eps_decay_steps"]),
        grad_clip       = float(cfg["dqn"]["grad_clip"]),
    )
    agent    = DqnAgent(dqn_cfg, device)
    start_ep = 1

    
    model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)

    if args.resume:
        ckpt_path = find_latest_checkpoint(model_dir)
        if ckpt_path is not None:
            print(f"[train.py] Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            agent.online_net.load_state_dict(ckpt["online"])
            agent.target_net.load_state_dict(ckpt["target"])
            agent.optim.load_state_dict(ckpt["optim"])
            agent.total_steps = ckpt.get("total_steps", 0)
            agent.load_buffer(ckpt_path)
            log_path = PROJECT_ROOT / "results" / "logs" / "dqn_log.jsonl"
            if "dqn_ep" in ckpt_path.stem:
                ckpt_ep = int(ckpt_path.stem.replace("dqn_ep", "")) + 1
                if log_path.exists():
                    with log_path.open() as f:
                        last = None
                        for line in f:
                            line = line.strip()
                            if line:
                                last = line
                    try:
                        log_ep = (json.loads(last)["episode"] + 1) if last else ckpt_ep
                    except (json.JSONDecodeError, KeyError, TypeError):
                        log_ep = ckpt_ep
                    start_ep = max(ckpt_ep, log_ep)
                else:
                    start_ep = ckpt_ep
            else:
                log_path = PROJECT_ROOT / "results" / "logs" / "dqn_log.jsonl"
                if log_path.exists():
                    with log_path.open() as f:
                        last = None
                        for line in f:
                            line = line.strip()
                            if line:
                                last = line
                    try:
                        start_ep = (json.loads(last)["episode"] + 1) if last else 1
                    except (json.JSONDecodeError, KeyError, TypeError):
                        start_ep = 1
                else:
                    start_ep = 1
            print(f"[train.py] Continuing from episode {start_ep} "
                  f"(total_steps={agent.total_steps}, eps={agent.epsilon():.3f})")
        else:
            print(f"[train.py] No checkpoint found — starting from scratch.")

    log_path = PROJECT_ROOT / "results" / "logs" / "dqn_log.jsonl"
    os.makedirs(log_path.parent, exist_ok=True)

    max_steps  = int(cfg["train"]["max_steps"])
    save_every = int(cfg["train"]["save_every"])

    print(f"[train.py] Connecting to NS-3 on port {args.port}...")
    obs, info = env.reset()
    print("[train.py] Connected. Starting training.")
    print(f"[train.py] obs[12:15] prb_align      = {obs[12]:.3f}, {obs[13]:.3f}, {obs[14]:.3f}")
    print(f"[train.py] obs[15:18] sla_prox/pelr/sla_prox = {obs[15]:.3f}, {obs[16]:.3f}, {obs[17]:.3f}")

    decoded = info.get("decoded_obs", None)

    
    reward_history: deque[float] = deque(maxlen=10)

    with TrainingMonitor("dqn", start_ep + args.episodes - 1, max_steps,
                         PROJECT_ROOT / "results") as mon:
        with log_path.open("a", encoding="utf-8") as logf:
            for ep in range(start_ep, start_ep + args.episodes):

                ep_reward  = 0.0
                step_count = 0
                sim_done   = False
                step_bar   = mon.begin_episode(ep)
                ep_losses: list[float] = []

                # Per-episode accumulators 
                # Throughput (Mbps)
                ep_embb_sum      = 0.0
                ep_urllc_thr_sum = 0.0
                ep_mmtc_sum      = 0.0

                # Latency (ms) 
                ep_embb_lat_sum  = 0.0;  ep_embb_lat_n  = 0
                ep_urllc_lat_sum = 0.0;  ep_urllc_lat_n = 0
                ep_mmtc_lat_sum  = 0.0;  ep_mmtc_lat_n  = 0

                # SLA rates (overall + per slice)
                ep_sla_sum       = 0.0
                ep_sla_embb_sum  = 0.0
                ep_sla_urllc_sum = 0.0
                ep_sla_mmtc_sum  = 0.0

                # HOL delay normalised [0,1] — from obs[9:12] or extra_json
                ep_hol_embb_sum  = 0.0
                ep_hol_urllc_sum = 0.0
                ep_hol_mmtc_sum  = 0.0

                # PRB efficiency normalised [0,1] — from obs[12:15]
                ep_eff_embb_sum  = 0.0
                ep_eff_urllc_sum = 0.0
                ep_eff_mmtc_sum  = 0.0

                # Average PRB allocation across episode
                ep_prb_embb_sum  = 0.0
                ep_prb_urllc_sum = 0.0
                ep_prb_mmtc_sum  = 0.0

                # Reward decomposition terms — from extra_json["reward_terms"]
                ep_rwd_thr_sum   = 0.0
                ep_rwd_sla_sum   = 0.0
                ep_rwd_eff_sum   = 0.0
                ep_rwd_viol_sum  = 0.0
                ep_active_sum    = 0.0

                # URLLC PELR — from extra_json["pkt_loss_rate"][1]
                ep_urllc_pelr_sum = 0.0
                

                for _ in range(max_steps):
                    action = agent.act(obs, explore=True)
                    next_obs, reward, done, truncated, info = env.step(action)

                    
                    if ep == start_ep and step_count == 0:
                        ns3_cfg = (info.get("extra_json") or {}).get("cfg", {})
                        if ns3_cfg:
                            for key, yaml_key in [
                                ("max_thr_mbps", "max_thr_mbps"),
                                ("min_thr_mbps", "min_thr_mbps"),
                                ("max_lat_ms",   "max_lat_ms"),
                            ]:
                                ns3_vals  = ns3_cfg.get(key, [])
                                yaml_vals = [float(cfg["env"][yaml_key][s]) for s in SLICE_NAMES]
                                if ns3_vals and (
                                    [round(v, 4) for v in ns3_vals] !=
                                    [round(v, 4) for v in yaml_vals]
                                ):
                                    raise RuntimeError(
                                        f"Config mismatch on '{key}':\n"
                                        f"  NS-3:        {ns3_vals}\n"
                                        f"  config.yaml: {yaml_vals}\n"
                                        f"  Edit one to match the other."
                                    )
                            print("[train.py] NS-3 / config.yaml consistency check passed.")
                        else:
                            print("[WARN] train.py: extra_json cfg absent — skipping check.")

                    terminal = bool(done or truncated)
                    agent.store(obs, action, reward, next_obs, terminal)
                    train_info = agent.train_step()
                    if isinstance(train_info, dict) and train_info.get("loss", 0.0) > 0.0:
                        ep_losses.append(train_info["loss"])

                    ep_reward  += float(reward)
                    step_count += 1
                    obs         = next_obs
                    decoded     = info.get("decoded_obs", decoded)

                    
                    if decoded is not None:
                        extra_json = info.get("extra_json")

                      
                        prb_frac_step = decoded.get("prb_frac", {})

                        ep_prb_embb_sum += (
                            float(prb_frac_step.get("eMBB", 0.0)) * _PRB_TOTAL
                        )
                        ep_prb_urllc_sum += (
                            float(prb_frac_step.get("URLLC", 0.0)) * _PRB_TOTAL
                        )
                        ep_prb_mmtc_sum += (
                            float(prb_frac_step.get("mMTC", 0.0)) * _PRB_TOTAL
                        )
                        
                        sla_rates = compute_sla_rates(decoded, cfg, extra_json)
                        ep_sla_sum       += sla_rates["overall"]
                        ep_sla_embb_sum  += sla_rates["embb"]
                        ep_sla_urllc_sum += sla_rates["urllc"]
                        ep_sla_mmtc_sum  += sla_rates["mmtc"]

                        #  Throughput (Mbps) 
                        ep_embb_sum      += (float(decoded["throughput"].get("eMBB",  0.0))
                                             * float(cfg["env"]["max_thr_mbps"]["eMBB"]))
                        ep_urllc_thr_sum += (float(decoded["throughput"].get("URLLC", 0.0))
                                             * float(cfg["env"]["max_thr_mbps"]["URLLC"]))
                        ep_mmtc_sum      += (float(decoded["throughput"].get("mMTC",  0.0))
                                             * float(cfg["env"]["max_thr_mbps"]["mMTC"]))

                        #  Latency (ms) 
                        lat_ms_raw = None
                        if extra_json and isinstance(extra_json.get("lat_ms"), list):
                            lat_ms_raw = extra_json["lat_ms"]

                        def _accum_lat(thr_key, lat_idx, lat_obs_key,
                                       lat_sum_ref, lat_n_ref, max_lat_key):
                            thr = (float(decoded["throughput"].get(thr_key, 0.0))
                                   * float(cfg["env"]["max_thr_mbps"][thr_key]))
                            if thr < 0.001:
                                return lat_sum_ref, lat_n_ref
                            if lat_ms_raw is not None:
                                return lat_sum_ref + float(lat_ms_raw[lat_idx]), lat_n_ref + 1
                            lat_norm = float(decoded["latency"].get(lat_obs_key, 0.0))
                            lat_ms   = lat_norm * 2.0 * float(cfg["env"]["max_lat_ms"][lat_obs_key])
                            return lat_sum_ref + lat_ms, lat_n_ref + 1

                        ep_embb_lat_sum,  ep_embb_lat_n  = _accum_lat(
                            "eMBB",  0, "eMBB",  ep_embb_lat_sum,  ep_embb_lat_n,  "eMBB")
                        ep_urllc_lat_sum, ep_urllc_lat_n = _accum_lat(
                            "URLLC", 1, "URLLC", ep_urllc_lat_sum, ep_urllc_lat_n, "URLLC")
                        ep_mmtc_lat_sum,  ep_mmtc_lat_n  = _accum_lat(
                            "mMTC",  2, "mMTC",  ep_mmtc_lat_sum,  ep_mmtc_lat_n,  "mMTC")

                        # HOL delay 
                        if extra_json and isinstance(extra_json.get("hol_norm"), list):
                            ep_hol_embb_sum  += float(extra_json["hol_norm"][0])
                            ep_hol_urllc_sum += float(extra_json["hol_norm"][1])
                            ep_hol_mmtc_sum  += float(extra_json["hol_norm"][2])
                        elif len(obs) >= 12:
                            ep_hol_embb_sum  += float(obs[9])
                            ep_hol_urllc_sum += float(obs[10])
                            ep_hol_mmtc_sum  += float(obs[11])

                        #  PRB efficiency 
                        if len(obs) >= 15:
                            ep_eff_embb_sum  += float(obs[12])
                            ep_eff_urllc_sum += float(obs[13])
                            ep_eff_mmtc_sum  += float(obs[14])

                        #  Reward decomposition 
                        rt = (extra_json or {}).get("reward_terms", {})
                        if rt:
                            ep_rwd_thr_sum  += float(rt.get("thr_norm_avg",    0.0))
                            ep_rwd_sla_sum  += float(rt.get("sla_margin_norm", 0.0))
                            ep_rwd_eff_sum  += float(rt.get("eff_norm",        0.0))
                            ep_rwd_viol_sum += float(rt.get("violation_rate",  0.0))
                            ep_active_sum   += float(rt.get("active_slices",   0.0))

                        #  URLLC PELR 
                        if extra_json and isinstance(extra_json.get("pkt_loss_rate"), list):
                            ep_urllc_pelr_sum += float(extra_json["pkt_loss_rate"][1])

                    mean_loss_so_far = float(np.mean(ep_losses)) if ep_losses else 0.0
                    mon.step(
                        step_idx    = step_count,
                        ep_reward   = ep_reward,
                        epsilon     = agent.epsilon(),
                        loss        = mean_loss_so_far if ep_losses else None,
                        buf_pct     = len(agent.buffer) / agent.cfg.buffer_size,
                        obs         = obs,
                        decoded_obs = decoded if isinstance(decoded, dict) else None,
                    )

                    if terminal:
                        sim_done = True
                        break

                step_bar.close()

                #  Episode average
                n = max(1, step_count)

                sla_rate  = ep_sla_sum       / n
                sla_embb  = ep_sla_embb_sum  / n
                sla_urllc = ep_sla_urllc_sum / n
                sla_mmtc  = ep_sla_mmtc_sum  / n

                embb_thr  = ep_embb_sum      / n
                urllc_thr = ep_urllc_thr_sum / n
                mmtc_thr  = ep_mmtc_sum      / n

                embb_lat  = (ep_embb_lat_sum  / ep_embb_lat_n)  if ep_embb_lat_n  > 0 else float("nan")
                urllc_lat = (ep_urllc_lat_sum / ep_urllc_lat_n) if ep_urllc_lat_n > 0 else float("nan")
                mmtc_lat  = (ep_mmtc_lat_sum  / ep_mmtc_lat_n)  if ep_mmtc_lat_n  > 0 else float("nan")

                hol_embb  = ep_hol_embb_sum  / n
                hol_urllc = ep_hol_urllc_sum / n
                hol_mmtc  = ep_hol_mmtc_sum  / n

                eff_embb  = ep_eff_embb_sum  / n
                eff_urllc = ep_eff_urllc_sum / n
                eff_mmtc  = ep_eff_mmtc_sum  / n

                avg_prb_embb  = ep_prb_embb_sum  / n
                avg_prb_urllc = ep_prb_urllc_sum / n
                avg_prb_mmtc  = ep_prb_mmtc_sum  / n

                rwd_thr_norm       = ep_rwd_thr_sum  / n
                rwd_sla_margin     = ep_rwd_sla_sum  / n
                rwd_eff_norm       = ep_rwd_eff_sum  / n
                rwd_violation_rate = ep_rwd_viol_sum / n
                active_slices_avg  = ep_active_sum   / n
                urllc_pelr_avg     = ep_urllc_pelr_sum / n

                mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

                # PRB allocation 
                if decoded is None:
                    decoded = {"prb_frac": {}}
                prb_frac  = decoded.get("prb_frac", {})

                
                reward_history.append(ep_reward)
                reward_mean10 = sum(reward_history) / len(reward_history)
                

                mon.end_episode(
                    ep_reward   = ep_reward,
                    sla_rate    = sla_rate,
                    embb_thr    = embb_thr,
                    urllc_lat   = urllc_lat,
                    decoded_obs = decoded,
                    epsilon     = agent.epsilon(),
                    mean_loss   = mean_loss,
                    buf_pct     = len(agent.buffer) / agent.cfg.buffer_size,
                    extra_metrics = {
                        # per-slice SLA
                        "sla/embb":        sla_embb,
                        "sla/urllc":       sla_urllc,
                        "sla/mmtc":        sla_mmtc,
                        # throughput
                        "throughput/URLLC_Mbps": urllc_thr,
                        "throughput/mMTC_Mbps":  mmtc_thr,
                        # latency (all slices)
                        "latency/eMBB_ms":  embb_lat  if embb_lat  == embb_lat  else 0.0,
                        "latency/mMTC_ms":  mmtc_lat  if mmtc_lat  == mmtc_lat  else 0.0,
                        # HOL delay
                        "hol/embb":   hol_embb,
                        "hol/urllc":  hol_urllc,
                        "hol/mmtc":   hol_mmtc,
                        # PRB efficiency
                        "efficiency/embb":  eff_embb,
                        "efficiency/urllc": eff_urllc,
                        "efficiency/mmtc":  eff_mmtc,
                        # reward decomposition
                        "reward_terms/thr_norm":       rwd_thr_norm,
                        "reward_terms/sla_margin":     rwd_sla_margin,
                        "reward_terms/eff_norm":       rwd_eff_norm,
                        "reward_terms/violation_rate": rwd_violation_rate,
                        "reward_terms/active_slices":  active_slices_avg,
                        # running mean
                        "reward/mean10": reward_mean10,
                    },
                )

                
                rec = {
                    
                    "episode":     ep,
                    "steps":       step_count,
                    # reward
                    "reward":      round(ep_reward,    6),
                    "reward_mean10": round(reward_mean10, 6),
                    # SLA
                    "sla_rate":    round(sla_rate,     6),
                    "sla_embb":    round(sla_embb,     6),
                    "sla_urllc":   round(sla_urllc,    6),
                    "sla_mmtc":    round(sla_mmtc,     6),
                    # throughput (Mbps)
                    "embb_thr":    round(embb_thr,     6),
                    "urllc_thr":   round(urllc_thr,    6),
                    "mmtc_thr":    round(mmtc_thr,     6),
                    # latency (ms) 
                    "embb_lat":    nan_or_round(embb_lat),
                    "urllc_lat":   nan_or_round(urllc_lat),
                    "mmtc_lat":    nan_or_round(mmtc_lat),
                    # HOL delay normalised [0,1]
                    "hol_embb":    round(hol_embb,     6),
                    "hol_urllc":   round(hol_urllc,    6),
                    "hol_mmtc":    round(hol_mmtc,     6),
                    # PRB allocation (last-step, integer PRBs)
                    "prb_embb": round(avg_prb_embb, 2),
                    "prb_urllc": round(avg_prb_urllc, 2),
                    "prb_mmtc": round(avg_prb_mmtc, 2),
                    # PRB efficiency normalised [0,1]
                    "eff_embb":    round(eff_embb,     6),
                    "eff_urllc":   round(eff_urllc,    6),
                    "eff_mmtc":    round(eff_mmtc,     6),
                    # reward decompositiom
                    "rwd_thr_norm":       round(rwd_thr_norm,       6),
                    "rwd_sla_margin":     round(rwd_sla_margin,     6),
                    "rwd_eff_norm":       round(rwd_eff_norm,       6),
                    "rwd_violation_rate": round(rwd_violation_rate, 6),
                    "active_slices_avg":  round(active_slices_avg,  6),
                    "urllc_pelr_avg":     round(urllc_pelr_avg,     8),
                    # training diagnostics
                    "epsilon":     round(agent.epsilon(), 6),
                    "mean_loss":   round(mean_loss,       6),
                    "buffer_size": len(agent.buffer),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()

                if ep % save_every == 0:
                    agent.save(model_dir / f"dqn_ep{ep}.pt")
                    print(f"[train.py] Checkpoint saved: dqn_ep{ep}.pt")

                if sim_done:
                    print(f"[train.py] NS-3 ended after episode {ep}. Increase SIM_TIME.")
                    break

    agent.save(model_dir / "dqn_final.pt")
    env.close()


if __name__ == "__main__":
    main()
