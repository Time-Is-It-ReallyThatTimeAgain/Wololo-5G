from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.slice_gym_env import SliceGymEnv, SLICE_NAMES
from envs.metrics import compute_sla_rates, nan_or_round

PolicyFn = Callable[[np.ndarray, Any], Tuple[int, Any]]

_ACTION_NO_CHANGE = 13  
_PRB_TOTAL        = 51
_PROGRESS_EVERY   = 50   




def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _action_from_delta(d_embb: int, d_urllc: int, d_mmtc: int) -> int:
    return (d_embb + 1) * 9 + (d_urllc + 1) * 3 + (d_mmtc + 1)


def _save(results: Dict, out_path: Path) -> None:
    """Incremental save — called after every policy so a crash loses nothing."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def _fmt_time(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))


def _print_global_eta(run_start: float, done: int, total: int) -> None:
    elapsed = time.time() - run_start
    if done >= total:
        print(f"\n[evaluate.py] All {total} polic{'y' if total == 1 else 'ies'} done in {_fmt_time(elapsed)}.")
    else:
        eta_s = (elapsed / done) * (total - done)
        print(f"\n[evaluate.py] Progress: {done}/{total} policies done "
              f"({_fmt_time(elapsed)} elapsed) — overall ETA: {_fmt_time(eta_s)}\n")


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

def random_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    return int(np.random.randint(0, 27)), hidden


def round_robin_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    """True Round-Robin: always drive toward equal PRB split 17/17/17.
    Takes the single action that most reduces the largest deficit."""
    target_frac  = np.array([17.0 / _PRB_TOTAL, 17.0 / _PRB_TOTAL, 17.0 / _PRB_TOTAL])
    current_frac = np.array([obs[0], obs[1], obs[2]])
    deficit      = target_frac - current_frac
    threshold    = 1.0 / _PRB_TOTAL  

    
    if np.all(np.abs(deficit) < threshold):
        return _ACTION_NO_CHANGE, hidden

    receiver = int(np.argmax(deficit))   
    donor    = int(np.argmin(deficit))   

    if receiver == donor:
        return _ACTION_NO_CHANGE, hidden

    delta = [0, 0, 0]
    delta[receiver] =  1
    delta[donor]    = -1
    return _action_from_delta(delta[0], delta[1], delta[2]), hidden


def greedy_pf_policy(obs: np.ndarray, hidden: Any) -> Tuple[int, Any]:
    """Proportional Fair: maximise thr[s] / avg_thr_history[s].
    hidden carries the exponential moving average of throughput per slice."""
   
    MAX_THR  = np.array([65.0, 30.0, 16.0])
    thr_mbps = np.array([obs[3], obs[4], obs[5]]) * MAX_THR

    
    if hidden is None:
        hidden = np.maximum(thr_mbps, 1e-3)

    avg_thr = hidden          
    alpha   = 0.1             

    active = thr_mbps > 1e-3
    if active.sum() < 2:
        
        hidden = (1 - alpha) * avg_thr + alpha * thr_mbps
        return _ACTION_NO_CHANGE, hidden

    
    pf_score = np.where(active, thr_mbps / np.maximum(avg_thr, 1e-9), 0.0)

    donor    = int(np.argmax(pf_score))                                      
    receiver = int(np.argmin(np.where(active, pf_score, np.inf)))            

   
    hidden = (1 - alpha) * avg_thr + alpha * thr_mbps

    if donor == receiver:
        return _ACTION_NO_CHANGE, hidden

    delta = [0, 0, 0]
    delta[donor]    = -1
    delta[receiver] =  1
    return _action_from_delta(delta[0], delta[1], delta[2]), hidden




def _accum_lat(
    slice_key: str,
    lat_idx: int,
    lat_sum: float,
    lat_n: int,
    decoded: Dict,
    lat_ms_raw: Any,
    cfg: Dict,
) -> Tuple[float, int]:
    thr = (float(decoded.get("throughput", {}).get(slice_key, 0.0))
           * float(cfg["env"]["max_thr_mbps"][slice_key]))
    if thr < 0.001:
        return lat_sum, lat_n
    if lat_ms_raw is not None:
        return lat_sum + float(lat_ms_raw[lat_idx]), lat_n + 1
    lat_norm = float(decoded.get("latency", {}).get(slice_key, 0.0))
    return lat_sum + lat_norm * 2.0 * float(cfg["env"]["max_lat_ms"][slice_key]), lat_n + 1




def evaluate_policy(
    name: str,
    policy_fn: PolicyFn,
    env: SliceGymEnv,
    obs: np.ndarray,
    episodes: int,
    max_steps: int,
    cfg: Dict,
    step_log_path: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict]:
    
    total_steps_planned = episodes * max_steps
    print(f"\n[evaluate.py] Evaluating {name} "
          f"({episodes} ep × {max_steps} steps = {total_steps_planned} total steps)...")

    hidden       = None
    policy_start = time.time()
    steps_done   = 0

    
    _step_log_fh = None
    if step_log_path is not None:
        step_log_path.parent.mkdir(parents=True, exist_ok=True)
        _step_log_fh = step_log_path.open("a", encoding="utf-8")

    ep_rewards:   List[float] = []
    ep_sla:       List[float] = []
    ep_sla_embb:  List[float] = []
    ep_sla_urllc: List[float] = []
    ep_sla_mmtc:  List[float] = []
    ep_embb_thrs: List[float] = []
    ep_urllc_thrs:List[float] = []
    ep_mmtc_thrs: List[float] = []
    ep_embb_lats: List[float] = []
    ep_urllc_lats:List[float] = []
    ep_mmtc_lats: List[float] = []
    ep_hol_embb:  List[float] = []
    ep_hol_urllc: List[float] = []
    ep_hol_mmtc:  List[float] = []
    ep_eff_embb:  List[float] = []
    ep_eff_urllc: List[float] = []
    ep_eff_mmtc:  List[float] = []
    ep_prb_embb:  List[float] = []
    ep_prb_urllc: List[float] = []
    ep_prb_mmtc:  List[float] = []
    ep_rwd_thr:   List[float] = []
    ep_rwd_sla:   List[float] = []
    ep_rwd_eff:   List[float] = []
    ep_rwd_viol:  List[float] = []
    ep_active:    List[float] = []

    for ep in range(episodes):
        ep_reward     = 0.0
        sla_sum       = 0.0
        sla_embb_sum  = 0.0
        sla_urllc_sum = 0.0
        sla_mmtc_sum  = 0.0
        embb_sum      = 0.0
        urllc_thr_sum = 0.0
        mmtc_sum      = 0.0
        embb_lat_sum  = 0.0; embb_lat_n  = 0
        urllc_lat_sum = 0.0; urllc_lat_n = 0
        mmtc_lat_sum  = 0.0; mmtc_lat_n  = 0
        hol_embb_sum  = 0.0
        hol_urllc_sum = 0.0
        hol_mmtc_sum  = 0.0
        eff_embb_sum  = 0.0
        eff_urllc_sum = 0.0
        eff_mmtc_sum  = 0.0
        prb_embb_sum  = 0.0
        prb_urllc_sum = 0.0
        prb_mmtc_sum  = 0.0
        rwd_thr_sum   = 0.0
        rwd_sla_sum   = 0.0
        rwd_eff_sum   = 0.0
        rwd_viol_sum  = 0.0
        active_sum    = 0.0
        step_count    = 0
        decoded: Dict = {}

        for step_i in range(max_steps):
            action, hidden = policy_fn(obs, hidden)
            obs, reward, done, truncated, info = env.step(action)
            decoded    = info.get("decoded_obs", decoded)
            ep_reward += float(reward)
            step_count += 1
            steps_done += 1

            extra_json = info.get("extra_json")

         
            _write_step = _step_log_fh is not None
            lat_ms_raw = extra_json.get("lat_ms") if extra_json else None

            # SLA
            rates = compute_sla_rates(decoded, cfg, extra_json)
            sla_sum       += rates["overall"]
            sla_embb_sum  += rates["embb"]
            sla_urllc_sum += rates["urllc"]
            sla_mmtc_sum  += rates["mmtc"]

            # Throughput
            embb_sum      += float(decoded.get("throughput", {}).get("eMBB",  0.0)) * float(cfg["env"]["max_thr_mbps"]["eMBB"])
            urllc_thr_sum += float(decoded.get("throughput", {}).get("URLLC", 0.0)) * float(cfg["env"]["max_thr_mbps"]["URLLC"])
            mmtc_sum      += float(decoded.get("throughput", {}).get("mMTC",  0.0)) * float(cfg["env"]["max_thr_mbps"]["mMTC"])

            # Latency
            embb_lat_sum,  embb_lat_n  = _accum_lat("eMBB",  0, embb_lat_sum,  embb_lat_n,  decoded, lat_ms_raw, cfg)
            urllc_lat_sum, urllc_lat_n = _accum_lat("URLLC", 1, urllc_lat_sum, urllc_lat_n, decoded, lat_ms_raw, cfg)
            mmtc_lat_sum,  mmtc_lat_n  = _accum_lat("mMTC",  2, mmtc_lat_sum,  mmtc_lat_n,  decoded, lat_ms_raw, cfg)

            # HOL delay — prefer extra_json, fall back to obs[9:12]
            if extra_json and isinstance(extra_json.get("hol_norm"), list):
                hol_embb_sum  += float(extra_json["hol_norm"][0])
                hol_urllc_sum += float(extra_json["hol_norm"][1])
                hol_mmtc_sum  += float(extra_json["hol_norm"][2])
            else:
                hol_embb_sum  += float(obs[9])
                hol_urllc_sum += float(obs[10])
                hol_mmtc_sum  += float(obs[11])

            # PRB efficiency obs[12:15]
            eff_embb_sum  += float(obs[12])
            eff_urllc_sum += float(obs[13])
            eff_mmtc_sum  += float(obs[14])

            # PRB allocation from decoded obs[0:3]
            prb_frac = decoded.get("prb_frac", {})
            prb_embb_sum  += prb_frac.get("eMBB",  0.0) * _PRB_TOTAL
            prb_urllc_sum += prb_frac.get("URLLC", 0.0) * _PRB_TOTAL
            prb_mmtc_sum  += prb_frac.get("mMTC",  0.0) * _PRB_TOTAL

            # Reward decomposition
            rt = (extra_json or {}).get("reward_terms", {})
            if rt:
                rwd_thr_sum  += float(rt.get("thr_norm_avg",    0.0))
                rwd_sla_sum  += float(rt.get("sla_margin_norm", 0.0))
                rwd_eff_sum  += float(rt.get("eff_norm",        0.0))
                rwd_viol_sum += float(rt.get("violation_rate",  0.0))
                active_sum   += float(rt.get("active_slices",   0.0))

            # write per-step record 
            if _write_step:
                prb_frac_s = decoded.get("prb_frac", {})
                rt_s       = (extra_json or {}).get("reward_terms", {})
                step_record = {
                    "policy":            name,
                    "episode":           ep + 1,
                    "step":              step_i + 1,
                    "reward":            round(float(reward), 6),
                    "sla_overall":       round(rates["overall"], 6),
                    "sla_embb":          round(rates["embb"],    6),
                    "sla_urllc":         round(rates["urllc"],   6),
                    "sla_mmtc":          round(rates["mmtc"],    6),
                    "embb_thr":          round(float(decoded.get("throughput", {}).get("eMBB",  0.0)) * float(cfg["env"]["max_thr_mbps"]["eMBB"]),  6),
                    "urllc_thr":         round(float(decoded.get("throughput", {}).get("URLLC", 0.0)) * float(cfg["env"]["max_thr_mbps"]["URLLC"]), 6),
                    "mmtc_thr":          round(float(decoded.get("throughput", {}).get("mMTC",  0.0)) * float(cfg["env"]["max_thr_mbps"]["mMTC"]),  6),
                    "embb_lat":          round((embb_lat_sum  / embb_lat_n)  if embb_lat_n  > 0 else float("nan"), 6),
                    "urllc_lat":         round((urllc_lat_sum / urllc_lat_n) if urllc_lat_n > 0 else float("nan"), 6),
                    "mmtc_lat":          round((mmtc_lat_sum  / mmtc_lat_n)  if mmtc_lat_n  > 0 else float("nan"), 6),
                    "hol_embb":          round(float(extra_json["hol_norm"][0]) if extra_json and isinstance(extra_json.get("hol_norm"), list) else float(obs[9]),  6),
                    "hol_urllc":         round(float(extra_json["hol_norm"][1]) if extra_json and isinstance(extra_json.get("hol_norm"), list) else float(obs[10]), 6),
                    "hol_mmtc":          round(float(extra_json["hol_norm"][2]) if extra_json and isinstance(extra_json.get("hol_norm"), list) else float(obs[11]), 6),
                    "eff_embb":          round(float(obs[12]), 6),
                    "eff_urllc":         round(float(obs[13]), 6),
                    "eff_mmtc":          round(float(obs[14]), 6),
                    "prb_embb":          round(prb_frac_s.get("eMBB",  0.0) * _PRB_TOTAL, 2),
                    "prb_urllc":         round(prb_frac_s.get("URLLC", 0.0) * _PRB_TOTAL, 2),
                    "prb_mmtc":          round(prb_frac_s.get("mMTC",  0.0) * _PRB_TOTAL, 2),
                    "rwd_thr_norm":      round(float(rt_s.get("thr_norm_avg",    0.0)), 6),
                    "rwd_sla_margin":    round(float(rt_s.get("sla_margin_norm", 0.0)), 6),
                    "rwd_eff_norm":      round(float(rt_s.get("eff_norm",        0.0)), 6),
                    "rwd_violation_rate":round(float(rt_s.get("violation_rate",  0.0)), 6),
                    "active_slices":     round(float(rt_s.get("active_slices",   0.0)), 6),
                }
                _step_log_fh.write(json.dumps(step_record) + "\n")
                _step_log_fh.flush()

            
            if (step_i + 1) % _PROGRESS_EVERY == 0:
                elapsed       = time.time() - policy_start
                secs_per_step = elapsed / max(1, steps_done)
                eta_s         = secs_per_step * (total_steps_planned - steps_done)
                n_so_far      = max(1, step_i + 1)
                print(f"    [{name}] ep {ep+1}/{episodes} "
                      f"step {step_i+1:>4}/{max_steps}  "
                      f"sla={sla_sum/n_so_far:.3f}  "
                      f"eMBB={embb_sum/n_so_far:.1f}Mbps  "
                      f"URLLC_lat={urllc_lat_sum/max(1,urllc_lat_n):.1f}ms  "
                      f"ETA: {_fmt_time(eta_s)}")

            if done or truncated:
                break

        n = max(1, step_count)

        ep_rewards.append(ep_reward)
        ep_sla.append(sla_sum / n)
        ep_sla_embb.append(sla_embb_sum  / n)
        ep_sla_urllc.append(sla_urllc_sum / n)
        ep_sla_mmtc.append(sla_mmtc_sum  / n)
        ep_embb_thrs.append(embb_sum      / n)
        ep_urllc_thrs.append(urllc_thr_sum / n)
        ep_mmtc_thrs.append(mmtc_sum      / n)
        ep_embb_lats.append((embb_lat_sum  / embb_lat_n)  if embb_lat_n  > 0 else float("nan"))
        ep_urllc_lats.append((urllc_lat_sum / urllc_lat_n) if urllc_lat_n > 0 else float("nan"))
        ep_mmtc_lats.append((mmtc_lat_sum  / mmtc_lat_n)  if mmtc_lat_n  > 0 else float("nan"))
        ep_hol_embb.append(hol_embb_sum  / n)
        ep_hol_urllc.append(hol_urllc_sum / n)
        ep_hol_mmtc.append(hol_mmtc_sum  / n)
        ep_eff_embb.append(eff_embb_sum  / n)
        ep_eff_urllc.append(eff_urllc_sum / n)
        ep_eff_mmtc.append(eff_mmtc_sum  / n)
        ep_prb_embb.append(prb_embb_sum  / n)
        ep_prb_urllc.append(prb_urllc_sum / n)
        ep_prb_mmtc.append(prb_mmtc_sum  / n)
        ep_rwd_thr.append(rwd_thr_sum  / n)
        ep_rwd_sla.append(rwd_sla_sum  / n)
        ep_rwd_eff.append(rwd_eff_sum  / n)
        ep_rwd_viol.append(rwd_viol_sum / n)
        ep_active.append(active_sum   / n)

        
        elapsed   = time.time() - policy_start
        eta_s     = (elapsed / steps_done) * (total_steps_planned - steps_done)
        print(f"  ✓ ep {ep+1:>2}/{episodes}  "
              f"reward={ep_reward:>8.3f}  "
              f"sla={sla_sum/n:.3f}  "
              f"eMBB_sla={sla_embb_sum/n:.3f}  "
              f"URLLC_sla={sla_urllc_sum/n:.3f}  "
              f"mMTC_sla={sla_mmtc_sum/n:.3f}  "
              f"ETA {name}: {_fmt_time(eta_s)}")

    if _step_log_fh is not None:
        _step_log_fh.close()

    def _mean(lst): return float(np.nanmean(lst))
    def _std(lst):  return float(np.nanstd(lst))

    results = {
        "mean_reward":             _mean(ep_rewards),
        "std_reward":              _std(ep_rewards),
        "mean_sla_rate":           _mean(ep_sla),
        "std_sla_rate":            _std(ep_sla),
        "mean_sla_embb":           _mean(ep_sla_embb),
        "std_sla_embb":            _std(ep_sla_embb),
        "mean_sla_urllc":          _mean(ep_sla_urllc),
        "std_sla_urllc":           _std(ep_sla_urllc),
        "mean_sla_mmtc":           _mean(ep_sla_mmtc),
        "std_sla_mmtc":            _std(ep_sla_mmtc),
        "mean_embb_thr_mbps":      _mean(ep_embb_thrs),
        "std_embb_thr_mbps":       _std(ep_embb_thrs),
        "mean_urllc_thr_mbps":     _mean(ep_urllc_thrs),
        "std_urllc_thr_mbps":      _std(ep_urllc_thrs),
        "mean_mmtc_thr_mbps":      _mean(ep_mmtc_thrs),
        "std_mmtc_thr_mbps":       _std(ep_mmtc_thrs),
        "mean_embb_lat_ms":        _mean(ep_embb_lats),
        "std_embb_lat_ms":         _std(ep_embb_lats),
        "mean_urllc_lat_ms":       _mean(ep_urllc_lats),
        "std_urllc_lat_ms":        _std(ep_urllc_lats),
        "mean_mmtc_lat_ms":        _mean(ep_mmtc_lats),
        "std_mmtc_lat_ms":         _std(ep_mmtc_lats),
        "mean_hol_embb":           _mean(ep_hol_embb),
        "mean_hol_urllc":          _mean(ep_hol_urllc),
        "mean_hol_mmtc":           _mean(ep_hol_mmtc),
        "mean_eff_embb":           _mean(ep_eff_embb),
        "mean_eff_urllc":          _mean(ep_eff_urllc),
        "mean_eff_mmtc":           _mean(ep_eff_mmtc),
        "mean_prb_embb":           _mean(ep_prb_embb),
        "std_prb_embb":            _std(ep_prb_embb),
        "mean_prb_urllc":          _mean(ep_prb_urllc),
        "std_prb_urllc":           _std(ep_prb_urllc),
        "mean_prb_mmtc":           _mean(ep_prb_mmtc),
        "std_prb_mmtc":            _std(ep_prb_mmtc),
        "mean_rwd_thr_norm":       _mean(ep_rwd_thr),
        "mean_rwd_sla_margin":     _mean(ep_rwd_sla),
        "mean_rwd_eff_norm":       _mean(ep_rwd_eff),
        "mean_rwd_violation_rate": _mean(ep_rwd_viol),
        "mean_active_slices":      _mean(ep_active),
        "episodes":                episodes,
    }
    return obs, results



# Main


def main() -> None:
    _VALID_POLICIES = ("random", "round-robin", "greedy-pf", "all")

    parser = argparse.ArgumentParser(
        description="Evaluate classical baselines against NS-3.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--port",      type=int, default=5555)
    parser.add_argument("--episodes",  type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--config",    type=str, default="configs/config.yaml")
    parser.add_argument("--out",       type=str, default="results/baseline_results.json")
    parser.add_argument("--step-log",  type=str, default=None,
                        help="Path for per-step JSONL log (one line per step). "
                             "Recommended with --episodes 1. Default: disabled.")
    parser.add_argument("--seed",      type=int, default=99)
    parser.add_argument(
        "--policy",
        type=str,
        default="all",
        choices=_VALID_POLICIES,
        help=(
            "Which policy to evaluate:\n"
            "  random      — uniform random action\n"
            "  round-robin — equal PRB split (17/17/17)\n"
            "  greedy-pf   — proportional fair heuristic\n"
            "  all         — run all three in sequence (default)"
        ),
    )
    args = parser.parse_args()

    cfg      = load_config(Path(args.config))
    out_path      = Path(args.out)
    step_log_path = Path(args.step_log) if args.step_log else None

    
    rng = np.random.default_rng(args.seed)

    def _random_policy(obs, hidden):
        return int(rng.integers(0, 27)), hidden

    _ALL_POLICIES: List[Tuple[str, PolicyFn]] = [
        ("Random",      _random_policy),
        ("Round-Robin", round_robin_policy),
        ("Greedy-PF",   greedy_pf_policy),
    ]

    _POLICY_MAP: Dict[str, List[Tuple[str, PolicyFn]]] = {
        "random":      [("Random",      _random_policy)],
        "round-robin": [("Round-Robin", round_robin_policy)],
        "greedy-pf":   [("Greedy-PF",   greedy_pf_policy)],
        "all":         _ALL_POLICIES,
    }

    selected_policies = _POLICY_MAP[args.policy]
    n_policies        = len(selected_policies)

    step_interval_s = cfg.get("env", {}).get("step_interval_ms", 200) / 1000.0
    total_steps     = n_policies * args.episodes * args.max_steps
    sim_s           = 3 * args.episodes * total_steps * step_interval_s * 10000

    print(f"[evaluate.py] Policy:     {args.policy}")
    print(f"[evaluate.py] Episodes:   {args.episodes} per policy")
    print(f"[evaluate.py] Steps:      {args.max_steps} per episode")
    print(f"[evaluate.py] Total steps:{total_steps} across {n_policies} polic{'y' if n_policies == 1 else 'ies'}")
    print(f"[evaluate.py] Step interval: {step_interval_s*1000:.0f}ms")
    print(f"[evaluate.py] NS-3 needs: simTime >= {sim_s:.0f}s")
    print(f"[evaluate.py] Output:     {out_path}")
    if step_log_path:
        print(f"[evaluate.py] Step log:   {step_log_path}  (per-step JSONL)")

    env = SliceGymEnv(port=args.port, sim_seed=args.seed, start_sim=False)
    print(f"\n[evaluate.py] Connecting to NS-3 on port {args.port}...")
    obs, _ = env.reset()
    print("[evaluate.py] Connected.\n")

    results:      Dict[str, Dict] = {}
    run_start     = time.time()
    policies_done = 0

    for name, policy_fn in selected_policies:
        obs, results[name] = evaluate_policy(
            name, policy_fn, env, obs, args.episodes, args.max_steps, cfg,
            step_log_path=step_log_path)
        _save(results, out_path)
        policies_done += 1
        _print_global_eta(run_start, policies_done, n_policies)

    env.close()

    
    col = 14
    print("\n" + "=" * 105)
    print(f"{'Policy':<{col}} {'Reward':>9} {'SLA%':>7} "
          f"{'eMBB_SLA':>9} {'URLLC_SLA':>10} {'mMTC_SLA':>9} "
          f"{'eMBB_Mbps':>10} {'URLLC_ms':>9} {'mMTC_ms':>8} {'ViolRate':>9}")
    print("-" * 105)
    for name, r in results.items():
        print(
            f"{name:<{col}}"
            f" {r['mean_reward']:>9.3f}"
            f" {r['mean_sla_rate']*100:>6.1f}%"
            f" {r['mean_sla_embb']*100:>8.1f}%"
            f" {r['mean_sla_urllc']*100:>9.1f}%"
            f" {r['mean_sla_mmtc']*100:>8.1f}%"
            f" {r['mean_embb_thr_mbps']:>10.2f}"
            f" {r['mean_urllc_lat_ms']:>9.3f}"
            f" {r['mean_mmtc_lat_ms']:>8.3f}"
            f" {r['mean_rwd_violation_rate']:>9.4f}"
        )
    print("=" * 105)
    print(f"\n[evaluate.py] Results saved → {out_path}")


if __name__ == "__main__":
    main()
