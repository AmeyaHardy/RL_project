"""SwarmSearch local results server.

Serves results.html plus a small JSON API that reads *everything* straight from the repo:
  - every evaluation file in res/*.json            (summaries, per-episode metrics, paired tests)
  - every training run in runs/*/                  (config.json + log CSV)
  - every GIF in the repo                          (viz/, project root, viz/generated/)
and generates new episode GIFs on demand from any checkpoint in runs/ (best.pt or latest.pt)
or any baseline policy, using swarm_search.visualize.

Everything runs locally: the server binds to 127.0.0.1 and the page loads no external assets.

    python viz_server.py            # http://127.0.0.1:8080 (opens your browser)
    python viz_server.py --port 9000 --no-browser
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import queue
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402

RES_DIR = PROJECT_ROOT / "res"
RUNS_DIR = PROJECT_ROOT / "runs"
VIZ_DIR = PROJECT_ROOT / "viz"
GEN_DIR = VIZ_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.json.sort_keys = False  # keep metric order as declared

# ── Metric metadata ──────────────────────────────────────────────────────────
# dir: +1 higher is better, -1 lower is better, 0 neutral
METRICS = {
    "return":            {"label": "Episode return",       "dir": 1,  "fmt": "f2"},
    "value":             {"label": "Delivered value",      "dir": 1,  "fmt": "f2"},
    "deliveries":        {"label": "Deliveries",           "dir": 1,  "fmt": "f2"},
    "persist_delivered": {"label": "Persistent delivered", "dir": 1,  "fmt": "f2"},
    "temp_delivered":    {"label": "Temp delivered",       "dir": 1,  "fmt": "f2"},
    "temp_capture":      {"label": "Temp capture rate",    "dir": 1,  "fmt": "pct"},
    "temp_expired":      {"label": "Temp expired",         "dir": -1, "fmt": "f2"},
    "collisions":        {"label": "Collisions",           "dir": -1, "fmt": "f2"},
    "malicious_picked":  {"label": "Malicious picked",     "dir": -1, "fmt": "f2"},
    "overlap":           {"label": "Redundant overlap",    "dir": -1, "fmt": "f3"},
    "coverage":          {"label": "Map coverage",         "dir": 1,  "fmt": "pct"},
    "mean_carrying":     {"label": "Mean carrying",        "dir": 0,  "fmt": "f2"},
    "length":            {"label": "Episode length",       "dir": 0,  "fmt": "f0"},
}

# ── What the page shows ─────────────────────────────────────────────────────
# Only the featured H-MAPPO checkpoint is shown. FEATURED_RESULTS maps each result file it produced to its
# run dir — verified 2026-11-15 by re-evaluating runs/hw4/best.pt and matching per-episode returns exactly.
# Baselines are shown only when evaluated on the same seeds and map as one of these results.
FEATURED_RUNS = ["hw4"]
FEATURED_RESULTS = {"hw4": "hw4", "t_hmappo": "hw4", "t5_zs_s0": "hw4"}

# Captions for pre-rendered GIFs (described by content, not file name); HIDDEN_GIFS are not from the featured model.
GIF_CAPTIONS = {
    "viz/hmappo_20000.gif": "H-MAPPO · test seed 20000",
    "viz/greedy_20000.gif": "Greedy · test seed 20000",
    "viz/scripted_20000.gif": "Scripted planner · test seed 20000",
    "shared_300.gif": "Shared-map comms · 300 steps",
    "window_300.gif": "Local-window comms · 300 steps",
    "private_300.gif": "No comms (private) · 300 steps",
}
HIDDEN_GIFS = {"viz/hmappo5_20000.gif"}

SPLIT_NAMES = {"val": "Validation", "test": "Test"}
COMM_NAMES = {"shared": "shared-map comms", "window": "local-window comms", "private": "no comms"}
BASELINE_NAMES = {
    "greedy": "Greedy", "scripted": "Scripted planner", "nearest_goal": "Nearest-goal planner",
    "random_goal": "Random-goal planner", "random": "Random",
}
BASELINE_POLICIES = ["greedy", "scripted", "nearest_goal", "random_goal", "random"]
INFO_MODES = ["shared", "window", "private"]
VIEWS = ["true", "agent", "both"]


def _clean(o):
    """Make data JSON-safe (NaN/inf -> None)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def _stages():
    from swarm_search.stages import STAGES
    from swarm_search.env import EnvConfig
    out = {}
    base = EnvConfig()
    for s, kw in STAGES.items():
        get = lambda k: kw.get(k, getattr(base, k))  # noqa: E731
        out[s] = {"height": get("height"), "width": get("width"), "n_agents": get("n_agents"),
                  "max_steps": get("max_steps"), "temp_spawn_prob": get("temp_spawn_prob"),
                  "n_malicious": get("n_malicious")}
    return out


def _infer_stage(cfg: dict):
    for s, st in _stages().items():
        if (cfg.get("height"), cfg.get("width")) == (st["height"], st["width"]):
            if s in (1, 2) and (cfg.get("temp_spawn_prob", 0) > 0) != (st["temp_spawn_prob"] > 0):
                continue
            return s
    return None


# ── Results (res/*.json) ─────────────────────────────────────────────────────
_res_cache: dict[str, tuple[float, dict]] = {}


def _load_result(stem: str) -> dict:
    p = RES_DIR / f"{stem}.json"
    if not p.is_file() or p.parent != RES_DIR:
        abort(404)
    mt = p.stat().st_mtime
    hit = _res_cache.get(stem)
    if hit and hit[0] == mt:
        return hit[1]
    d = json.loads(p.read_text())
    _res_cache[stem] = (mt, d)
    return d


def _map_desc(cfg: dict) -> str:
    n = cfg.get("n_agents")
    return f"{cfg.get('height')}×{cfg.get('width')} map · {n} robot{'' if n == 1 else 's'}"


def _run_env(run: str) -> dict:
    p = RUNS_DIR / run / "config.json"
    return json.loads(p.read_text()).get("env_cfg", {}) if p.is_file() else {}


def _pair_key(d: dict):
    """Two results can be paired only if they used the same seeds on the same map size and robot count."""
    c = d.get("env_cfg", {})
    return (tuple(d.get("seeds", [])), c.get("height"), c.get("width"), c.get("n_agents"))


def included_stems() -> list[str]:
    stems = {p.stem for p in RES_DIR.glob("*.json")}
    feat = [s for s in FEATURED_RESULTS if s in stems]
    keys = {_pair_key(_load_result(s)) for s in feat}
    base = [s for s in stems - set(feat)
            if not _load_result(s).get("label", "").endswith(".pt") and _pair_key(_load_result(s)) in keys]
    return feat + sorted(base)


def _included(stem: str) -> dict:
    if stem not in included_stems():
        abort(404)
    return _load_result(stem)


def result_entry(stem: str) -> dict:
    d = _load_result(stem)
    cfg = d.get("env_cfg", {})
    label = d.get("label", "")
    trained = label.endswith(".pt")
    run = FEATURED_RESULTS.get(stem) if trained else None
    comm = COMM_NAMES.get(cfg.get("info_mode"), cfg.get("info_mode"))
    if trained:
        name = f"H-MAPPO · {comm}"
        rc = _run_env(run) if run else {}
        if rc and (rc.get("height"), rc.get("n_agents")) != (cfg.get("height"), cfg.get("n_agents")):
            name += f" · zero-shot (trained on {rc.get('height')}×{rc.get('width')}, {rc.get('n_agents')} robots)"
    else:
        name = f"{BASELINE_NAMES.get(label, label)} · {comm}"
    seeds = d.get("seeds", [])
    split = d.get("split", "?")
    split_name = SPLIT_NAMES.get(split, split)
    return {
        "file": stem, "name": name, "label": label, "trained": trained, "run": run,
        "policy": "hmappo" if trained else label, "split": split, "variant": d.get("variant", "default"),
        "n": len(seeds), "seed_first": seeds[0] if seeds else None, "seed_last": seeds[-1] if seeds else None,
        "seeds_hash": hashlib.md5(json.dumps(_pair_key(d)).encode()).hexdigest()[:10],
        "grid": f"{cfg.get('height')}×{cfg.get('width')}", "n_agents": cfg.get("n_agents"),
        "info_mode": cfg.get("info_mode"), "max_steps": cfg.get("max_steps"), "stage": _infer_stage(cfg),
        "setting": f"{split_name} · {_map_desc(cfg)} · {len(seeds)} episodes",
        "short_setting": f"{split_name}, {cfg.get('height')}×{cfg.get('width')}",
        "summary": d.get("summary", {}),
    }


@app.route("/api/results")
def api_results():
    return jsonify(_clean({"metrics": METRICS, "results": [result_entry(s) for s in included_stems()]}))


@app.route("/api/results/<stem>")
def api_result(stem):
    d = _included(stem)
    return jsonify(_clean({"entry": result_entry(stem), "seeds": d.get("seeds", []),
                           "metrics": d.get("metrics", {}), "env_cfg": d.get("env_cfg", {})}))


@app.route("/api/compare")
def api_compare():
    a_stem, b_stem = request.args.get("a", ""), request.args.get("b", "")
    a, b = _included(a_stem), _included(b_stem)
    if _pair_key(a) != _pair_key(b):
        return jsonify({"error": "These two results were not evaluated on the same seeds and map, so a paired test is not possible."}), 400
    from swarm_search.evaluate import paired_compare
    rows = {}
    for m in METRICS:
        if m in a["metrics"] and m in b["metrics"]:
            rows[m] = paired_compare(a, b, metric=m)
            rows[m]["a"] = a["metrics"][m]
            rows[m]["b"] = b["metrics"][m]
    return jsonify(_clean({"a": result_entry(a_stem), "b": result_entry(b_stem), "seeds": a["seeds"], "rows": rows}))


# ── Training runs (runs/*) ──────────────────────────────────────────────────
def _read_log(path: Path) -> dict:
    cols: dict[str, list] = {}
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        for k in rd.fieldnames or []:
            cols[k] = []
        for row in rd:
            for k in cols:
                try:
                    v = float(row.get(k, "nan"))
                    cols[k].append(v if math.isfinite(v) else None)
                except (TypeError, ValueError):
                    cols[k].append(None)
    return cols


def _run_display(env: dict) -> str:
    return f"H-MAPPO · {COMM_NAMES.get(env.get('info_mode'), env.get('info_mode'))} · {_map_desc(env)}"


def _public_config(cfg: dict) -> dict:
    """Config for display: drop file paths / bookkeeping, describe the warm start by its features."""
    cfg = json.loads(json.dumps(cfg))
    args = cfg.get("args", {})
    for k in ("out", "ref", "device", "selftest", "map_pool"):
        args.pop(k, None)
    init = args.pop("init_actor_from", None)
    if init:
        env = _run_env(Path(init).parent.name)
        args["warm_start"] = f"actor from an H-MAPPO policy trained on {_map_desc(env)}" if env else "pretrained actor"
    else:
        args["warm_start"] = "none (trained from scratch)"
    return cfg


def discover_runs() -> list[dict]:
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or d.name not in FEATURED_RUNS:
            continue
        ckpts = [c for c in ("best.pt", "latest.pt") if (d / c).is_file()]
        if not ckpts:
            continue
        cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").is_file() else {}
        env = cfg.get("env_cfg", {})
        args = cfg.get("args", {})
        logs = sorted(d.glob("log*.csv"))
        runs.append({
            "name": d.name, "checkpoints": ckpts, "selftest": d.name.startswith("_") or bool(args.get("selftest")),
            "stage": args.get("stage"), "info_mode": env.get("info_mode", args.get("info_mode")),
            "grid": f"{env.get('height')}×{env.get('width')}", "n_agents": env.get("n_agents"),
            "max_steps": env.get("max_steps"), "seed": args.get("seed"), "total_steps": args.get("total_steps"),
            "init_from": args.get("init_actor_from"), "log_file": logs[0].name if logs else None,
            "display": _run_display(env), "config": _public_config(cfg),
        })
    return runs


def _run_dir(name: str) -> Path:
    for r in discover_runs():
        if r["name"] == name:
            return RUNS_DIR / name
    abort(404)


@app.route("/api/runs")
def api_runs():
    runs = discover_runs()
    for r in runs:
        r["log"] = _read_log(RUNS_DIR / r["name"] / r["log_file"]) if r["log_file"] else {}
    return jsonify(_clean({"runs": runs}))


# ── GIF gallery ─────────────────────────────────────────────────────────────
def _gif_meta(path: Path) -> dict:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    meta = {"name": path.name, "url": "/gif/" + rel, "path": rel, "size": path.stat().st_size,
            "mtime": path.stat().st_mtime, "generated": path.parent == GEN_DIR,
            "caption": GIF_CAPTIONS.get(rel, "Pre-rendered episode")}
    side = path.with_suffix(".json")
    if side.is_file():
        try:
            meta["params"] = json.loads(side.read_text())
        except Exception:
            pass
    return meta


def list_gifs() -> list[dict]:
    paths = list(PROJECT_ROOT.glob("*.gif")) + list(VIZ_DIR.glob("*.gif")) + list(GEN_DIR.glob("*.gif"))
    paths = [p for p in paths if p.relative_to(PROJECT_ROOT).as_posix() not in HIDDEN_GIFS and not p.name.endswith(".part.gif")]
    return sorted((_gif_meta(p) for p in paths), key=lambda m: (m["generated"], -m["mtime"] if m["generated"] else 0, m["path"]))


@app.route("/api/gifs")
def api_gifs():
    return jsonify({"gifs": list_gifs()})


@app.route("/gif/<path:rel>")
def serve_gif(rel):
    p = (PROJECT_ROOT / rel).resolve()
    allowed = {PROJECT_ROOT, VIZ_DIR.resolve(), GEN_DIR.resolve()}
    if p.suffix.lower() != ".gif" or p.parent not in allowed or not p.is_file():
        abort(404)
    return send_from_directory(str(p.parent), p.name, max_age=0)


@app.route("/api/gifs/generated/<name>", methods=["DELETE"])
def delete_gif(name):
    p = (GEN_DIR / name).resolve()
    if p.parent != GEN_DIR.resolve() or p.suffix != ".gif" or not p.is_file():
        abort(404)
    p.unlink()
    p.with_suffix(".json").unlink(missing_ok=True)
    return jsonify({"ok": True})


# ── On-demand GIF generation ────────────────────────────────────────────────
@app.route("/api/options")
def api_options():
    from swarm_search.stages import VARIANTS
    runs = [{k: r[k] for k in ("name", "display", "checkpoints", "stage", "info_mode", "grid", "n_agents", "max_steps", "selftest")}
            for r in discover_runs()]
    return jsonify({"runs": runs, "baselines": [{"id": b, "name": BASELINE_NAMES[b]} for b in BASELINE_POLICIES],
                    "stages": _stages(), "info_modes": INFO_MODES, "variants": list(VARIANTS), "views": VIEWS})


def _int(d, k, default, lo, hi):
    try:
        v = int(d.get(k, default) if d.get(k) not in (None, "") else default)
    except (TypeError, ValueError):
        raise ValueError(f"'{k}' must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"'{k}' must be between {lo} and {hi}")
    return v


def _result_stem(d: dict) -> str:
    stem = str(d.get("result", ""))
    if stem not in included_stems():
        raise ValueError("Unknown result")
    return stem


def normalize_params(d: dict) -> dict:
    """Validate a generation request and return a canonical parameter dict."""
    from swarm_search.stages import STAGES, VARIANTS
    p: dict = {}
    source = d.get("source", "checkpoint")
    if source == "checkpoint":
        run = str(d.get("run", ""))
        r = next((r for r in discover_runs() if r["name"] == run), None)
        if r is None:
            raise ValueError(f"Unknown run '{run}'")
        ckpt = d.get("ckpt", "best.pt")
        if ckpt not in r["checkpoints"]:
            raise ValueError(f"{run} has no {ckpt}")
        p.update(source="checkpoint", policy="hmappo", run=run, ckpt=ckpt)
        env = d.get("env", "checkpoint")
        if env in ("checkpoint", "", None):
            p["env"] = "checkpoint"
        elif env == "result":
            p.update(env="result", result=_result_stem(d))
        else:
            p["env"] = "stage"
            p["stage"] = _int(d, "stage", 4, 1, 5)
    elif source == "baseline":
        pol = d.get("policy")
        if pol not in BASELINE_POLICIES:
            raise ValueError(f"Unknown baseline '{pol}'")
        p.update(source="baseline", policy=pol)
        if d.get("env") == "result":
            p.update(env="result", result=_result_stem(d))
        else:
            p.update(env="stage", stage=_int(d, "stage", 4, 1, 5), info_mode=d.get("info_mode", "shared"))
            if p["info_mode"] not in INFO_MODES:
                raise ValueError("bad info_mode")
        p["memory"] = bool(d.get("memory", True))
    else:
        raise ValueError("source must be 'checkpoint' or 'baseline'")
    if p.get("env") == "stage" and p["stage"] not in STAGES:
        raise ValueError("bad stage")
    variant = d.get("variant", "default") or "default"
    if variant not in VARIANTS:
        raise ValueError("bad variant")
    p["variant"] = variant
    p["seed"] = _int(d, "seed", 20000, 0, 2**31 - 1)
    p["steps"] = _int(d, "steps", 0, 0, 2000)  # 0 = env default
    p["fps"] = _int(d, "fps", 10, 1, 30)
    p["fog"] = bool(d.get("fog", True))
    p["view"] = d.get("view", "true") if d.get("view", "true") in VIEWS else "true"
    p["agent"] = _int(d, "agent", 0, 0, 15)
    p["goals"] = bool(d.get("goals", True))
    return p


def params_key(p: dict) -> str:
    return hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]


def params_title(p: dict) -> str:
    if p["source"] == "checkpoint":
        who = f"H-MAPPO ({'best' if p['ckpt'] == 'best.pt' else 'final'} weights)"
    else:
        who = BASELINE_NAMES[p["policy"]]
    if p["env"] == "checkpoint":
        env = f"training env ({_map_desc(_run_env(p['run']))})"
    elif p["env"] == "result":
        env = result_entry(p["result"])["setting"].rsplit(" · ", 1)[0] + " env"
    else:
        st = _stages()[p["stage"]]
        env = _map_desc(st)
        if p["source"] == "baseline":
            env += f" · {COMM_NAMES.get(p['info_mode'], p['info_mode'])}"
    if p["variant"] != "default":
        env += f" · {p['variant']} variant"
    return f"{who} · {env} · seed {p['seed']}"


class _ProgressPolicy:
    """Wraps a policy callable to report step progress; delegates `overlay` etc. to the wrapped policy."""

    def __init__(self, pol, env, job, steps):
        self._pol, self._env, self._job, self._steps = pol, env, job, max(1, steps)

    def __call__(self, obs):
        self._job["progress"] = min(0.99, self._env.t / self._steps)
        return self._pol(obs)

    def __getattr__(self, name):
        return getattr(self._pol, name)


def build_and_render(p: dict, out_path: Path, job: dict):
    import matplotlib
    matplotlib.use("Agg")
    from swarm_search.env import SwarmSearchEnv
    from swarm_search.stages import VARIANTS, cfg_from_dict, make_env_config, cfg_to_dict
    from swarm_search.visualize import make_policy, stream_to_gif

    ckpt_path = None
    if p["source"] == "checkpoint":
        ckpt_path = str(RUNS_DIR / p["run"] / p["ckpt"])
    if p["env"] == "result":
        cfg = cfg_from_dict(_load_result(p["result"])["env_cfg"], variant=p["variant"])
    elif p["source"] == "checkpoint":
        import torch
        ckpt_path = str(RUNS_DIR / p["run"] / p["ckpt"])
        if p["env"] == "checkpoint":
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)["env_cfg"]
            cfg = cfg_from_dict(raw, variant=p["variant"])
        else:
            info = json.loads((RUNS_DIR / p["run"] / "config.json").read_text())["env_cfg"].get("info_mode", "window") \
                if (RUNS_DIR / p["run"] / "config.json").is_file() else "window"
            cfg = cfg_from_dict(cfg_to_dict(make_env_config(p["stage"], info)), variant=p["variant"])
    else:
        cfg = make_env_config(p["stage"], p["info_mode"], **VARIANTS[p["variant"]])
    if p["steps"]:
        cfg.max_steps = p["steps"]
    if p["agent"] >= cfg.n_agents:
        p["agent"] = 0
    job["status"] = "running"
    job["detail"] = f"{cfg.height}×{cfg.width}, {cfg.n_agents} robots, {cfg.info_mode}, {cfg.max_steps} steps"
    env = SwarmSearchEnv(cfg, render_mode="rgb_array")
    pol = make_policy(p["policy"], env, seed=p["seed"], checkpoint=ckpt_path, memory=p.get("memory", True))
    tmp = out_path.with_name(out_path.stem + ".part.gif")
    stream_to_gif(env, _ProgressPolicy(pol, env, job, cfg.max_steps), p["seed"], cfg.max_steps, p["fog"], p["fps"],
                  str(tmp), view=p["view"], agent=p["agent"], show_goals=p["goals"])
    tmp.replace(out_path)
    s = env.stats
    return {"t": env.t, "deliveries": s.get("deliveries"), "return": s.get("ep_return"), "env": job["detail"]}


JOBS: dict[str, dict] = {}
JOB_Q: "queue.Queue[str]" = queue.Queue()


def _worker():
    while True:
        jid = JOB_Q.get()
        job = JOBS[jid]
        p = job["params"]
        out = GEN_DIR / job["file"]
        job["status"], job["started"] = "loading", time.time()
        try:
            stats = build_and_render(p, out, job)
            meta = {"params": p, "title": job["title"], "stats": stats, "created": time.time(),
                    "seconds": round(time.time() - job["started"], 1)}
            out.with_suffix(".json").write_text(json.dumps(_clean(meta), indent=1))
            job.update(status="done", progress=1.0, url=f"/gif/viz/generated/{job['file']}", stats=_clean(stats),
                       seconds=meta["seconds"])
        except Exception as e:  # report to the page
            traceback.print_exc()
            job.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            JOB_Q.task_done()


threading.Thread(target=_worker, daemon=True, name="gif-worker").start()


def _job_view(j):
    return {k: j.get(k) for k in ("id", "status", "progress", "title", "url", "error", "detail", "stats",
                                  "seconds", "cached", "queued_at", "params")} | {"position": _queue_pos(j)}


def _queue_pos(j):
    if j.get("status") != "queued":
        return 0
    return sum(1 for o in JOBS.values() if o["status"] == "queued" and o["queued_at"] <= j["queued_at"])


@app.route("/api/generate", methods=["POST"])
def api_generate():
    try:
        p = normalize_params(request.get_json(force=True) or {})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    key = params_key(p)
    fname = f"{p['policy']}_{p.get('run') or p.get('result') or 's' + str(p.get('stage'))}_{p['seed']}_{key}.gif"
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "params": p, "title": params_title(p), "file": fname, "progress": 0.0,
           "queued_at": time.time(), "cached": False}
    out = GEN_DIR / fname
    # identical request already running/queued -> reuse it
    for o in JOBS.values():
        if o["file"] == fname and o["status"] in ("queued", "loading", "running"):
            return jsonify(_job_view(o))
    JOBS[jid] = job
    if out.is_file():
        side = out.with_suffix(".json")
        stats = json.loads(side.read_text()).get("stats") if side.is_file() else None
        job.update(status="done", progress=1.0, url=f"/gif/viz/generated/{fname}", cached=True, stats=stats)
    else:
        job["status"] = "queued"
        JOB_Q.put(jid)
    return jsonify(_job_view(job))


@app.route("/api/jobs/<jid>")
def api_job(jid):
    j = JOBS.get(jid)
    if not j:
        abort(404)
    return jsonify(_job_view(j))


# ── Page ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(PROJECT_ROOT), "results.html", max_age=0)


def main():
    ap = argparse.ArgumentParser(description="Local SwarmSearch results page + on-demand GIF generator")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    print(f"SwarmSearch results — {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
