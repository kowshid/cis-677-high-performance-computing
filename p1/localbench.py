#!/usr/bin/env python3
"""localbench.py -- measure every entry of an hpcbench workdir on this machine,
the way `hpcbench run` measures it, and write the same kind of bundle.
Nothing is posted anywhere, and nothing in your entry folders is modified.

Run it from the folder `hpcbench pull cis677-p1 --out p1` created:

    cd p1
    python3 localbench.py                          # every entry: 1 warmup + 5 timed, 4 threads
    python3 localbench.py --only ahsan-07 ahsan-08 # baseline is always included
    python3 localbench.py --runs 11                # steadier medians

For each entry, following the manifest's own contract:

    build     sh ./build.sh                                in a scratch copy under .localbench/
    prepare   ./build/task prepare <matrix.bin> <work>     untimed
    solve     ./build/task solve <work> <result> <k>       per k: warmups, then timed runs

  * The time of a run is the harness's own HPCBENCH_TIME_MS -- the timed solve
    only -- the same number hpcbench reports.
  * Each result is checked with the grader's canonical digest (quantize to the
    task's significant digits, SHA-256; the logic of hpcbench/canon.py in the
    course repository). A digest that differs from the manifest's reference is
    then checked against the task's tolerance (relative difference over the
    whole answer), using this session's baseline result.
  * cpu_efficiency, user_s and sys_s describe the timed runs of the LAST sweep
    value, as in hpcbench's bundles; cpu_efficiency = process CPU time /
    (timed ms x threads).
  * peak_rss_mb is this entry's own peak over its solve runs. (hpcbench's
    bundles carry a session-wide high-water mark there, which is why every
    entry after the first shows the same 381.92 MB.)
  * Solve runs get OMP_NUM_THREADS=<threads>, OMP_PROC_BIND=close and
    OMP_PLACES=cores, as the course runner sets them.

Needs only python3 (3.9+); make_p1_input.py (needs numpy) runs only if
matrix.bin has to be rebuilt.
"""
import argparse
import datetime
import filecmp
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

# ---- the grader's canonical digest (hpcbench/canon.py, same logic) ---------------------

CANON_VERSION = 1
MAX_EXACT_POW10 = 22


def quantize(x, sig):
    if not math.isfinite(x):
        raise ValueError(f"non-finite value in result: {x!r}")
    if x == 0.0 or abs(x) < 1e-300:
        return 0.0
    k = sig - 1 - math.floor(math.log10(abs(x)))
    if abs(k) > MAX_EXACT_POW10:
        raise ValueError(f"cannot canonicalize {x!r} at {sig} significant digits")
    if k >= 0:
        factor = 10.0 ** k
        q = round(x * factor) / factor
    else:
        inv = 10.0 ** (-k)
        q = round(x / inv) * inv
    return 0.0 if q == 0.0 else q


def read_result(path):
    with open(path, "rb") as f:
        if f.read(8) != b"HPCBENCH":
            raise ValueError(f"{path}: not an hpcbench result file")
        version, ndim = struct.unpack("<II", f.read(8))
        if version != CANON_VERSION:
            raise ValueError(f"{path}: canon version {version}, expected {CANON_VERSION}")
        dims = struct.unpack(f"<{ndim}Q", f.read(8 * ndim)) if ndim else ()
        (n,) = struct.unpack("<Q", f.read(8))
        raw = f.read(8 * n)
        if len(raw) != 8 * n or f.read(1):
            raise ValueError(f"{path}: truncated, or trailing bytes")
    return dims, struct.unpack(f"<{n}d", raw)


def canon_digest(path, sig):
    dims, values = read_result(path)
    h = hashlib.sha256()
    h.update(b"hpcbench-canon-v%d\0" % CANON_VERSION)
    h.update(b"sig%d\0" % sig)
    h.update(b"shape:" + b",".join(b"%d" % d for d in dims) + b"\0")
    h.update(b"n:%d\0" % len(values))
    h.update(struct.pack(f"<{len(values)}d", *(quantize(v, sig) for v in values)))
    return h.hexdigest(), values


def relative_error(values, ref):
    """||Y - Yref|| / ||Yref|| over the whole answer."""
    if len(values) != len(ref):
        return math.inf
    num = math.fsum((a - b) * (a - b) for a, b in zip(values, ref))
    den = math.fsum(b * b for b in ref)
    return math.sqrt(num / den) if den > 0 else (0.0 if num == 0 else math.inf)


# ---- running one process, with its own CPU time and peak memory -------------------------

def run_once(cmd, cwd, env, timeout):
    with tempfile.TemporaryFile() as fo, tempfile.TemporaryFile() as fe:
        t0 = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fo, stderr=fe)
        killed = threading.Event()

        def kill():
            killed.set()
            proc.kill()

        timer = threading.Timer(timeout, kill)
        timer.start()
        try:
            _, status, ru = os.wait4(proc.pid, 0)  # this child's own rusage
        finally:
            timer.cancel()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        code = 124 if killed.is_set() else os.waitstatus_to_exitcode(status)
        proc.returncode = code  # already reaped; keep Popen from waiting again
        fo.seek(0)
        fe.seek(0)
        out = fo.read().decode(errors="replace")
        err = fe.read().decode(errors="replace")
    rss_mb = ru.ru_maxrss / (1024.0 * 1024.0) if sys.platform == "darwin" else ru.ru_maxrss / 1024.0
    return {"code": code, "out": out, "err": err, "user": ru.ru_utime, "sys": ru.ru_stime,
            "rss_mb": rss_mb, "wall_ms": wall_ms}


def timed_ms(stdout):
    for line in stdout.splitlines():
        if line.startswith("HPCBENCH_TIME_MS"):
            return float(line.split()[1])
    return None


# ---- machine description ----------------------------------------------------------------

def sh_out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def device_info(cxx):
    if sys.platform == "darwin":
        cpu = sh_out(["sysctl", "-n", "machdep.cpu.brand_string"])
        phys = int(sh_out(["sysctl", "-n", "hw.physicalcpu"]) or 0)
        logical = int(sh_out(["sysctl", "-n", "hw.logicalcpu"]) or 0)
        mem = int(sh_out(["sysctl", "-n", "hw.memsize"]) or 0) / 2 ** 30
    else:
        cpu, cores, mem = platform.processor(), set(), 0.0
        try:
            block = {}
            for line in open("/proc/cpuinfo"):
                if ":" in line:
                    key, val = (s.strip() for s in line.split(":", 1))
                    block[key] = val
                    if key == "model name":
                        cpu = val
                elif block:
                    cores.add((block.get("physical id"), block.get("core id")))
                    block = {}
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal:"):
                    mem = int(line.split()[1]) / 2 ** 20
        except OSError:
            pass
        logical = os.cpu_count() or 0
        phys = len(cores) if cores and None not in next(iter(cores)) else logical
    compiler = (sh_out([cxx, "--version"]).splitlines() or [""])[0]
    return {"cpu": cpu, "arch": platform.machine(), "os": f"{platform.system()} {platform.release()}",
            "compiler": compiler, "cores_physical": phys, "cores_logical": logical,
            "memory_gb": round(mem, 1)}


def links_openmp(binary):
    if not Path(binary).is_file():
        return None
    out = sh_out(["otool", "-L", str(binary)] if sys.platform == "darwin" else ["ldd", str(binary)])
    return any(lib in out for lib in ("libomp", "libgomp", "libiomp"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- the input ------------------------------------------------------------------------------

def ensure_input(wd, manifest, say):
    raw = wd / "matrix.bin"
    npz = wd / "cis677-p1-input.bin"  # where hpcbench keeps the downloaded packed input
    inp = manifest.get("input", {})
    if not raw.is_file():
        if not npz.is_file():
            say("input", f"downloading {inp['url']}")
            urllib.request.urlretrieve(inp["url"], npz)
        script = wd / "make_p1_input.py"
        if not script.is_file():
            url = re.search(r"https://\S+make_p1_input\.py", inp.get("prepare_cmd", ""))
            if not url:
                sys.exit("make_p1_input.py is missing and the manifest does not say where it is")
            urllib.request.urlretrieve(url.group(0), script)
        say("input", "preparing matrix.bin with make_p1_input.py")
        subprocess.run([sys.executable, str(script), "--packed", str(npz), "--variant", "public",
                        "--out", str(raw)], check=True)
    info = {"mode": "local", "sha256": None, "prepared": True}
    if npz.is_file() and npz.stat().st_size < (64 << 20):
        digest = sha256_file(npz)
        info = {"mode": "url" if digest == inp.get("sha256") else "local", "sha256": digest,
                "prepared": True}
    return raw, info


# ---- one entry --------------------------------------------------------------------------------

def measure_entry(name, a, ctx):
    src, scratch = ctx["wd"] / name, ctx["root"] / name
    shutil.rmtree(scratch, ignore_errors=True)
    shutil.copytree(src, scratch, ignore=shutil.ignore_patterns(
        "build", "*.bin", ".hpcbench-dims", ".DS_Store", "__pycache__"))
    shell = shutil.which("bash") or "sh"
    rec = {"submission": name, "status": "ok", "per_sweep": [], "peak_rss_mb": 0.0,
           "cpu_efficiency": 0.0,
           "profile": {"build_cmd": f"{os.path.basename(shell)} {src / 'build.sh'}", "user_s": 0.0,
                       "sys_s": 0.0, "error": None, "entry_version": None,
                       "selected_because": ctx["reason"](name)}}

    def fail(status, msg):
        rec["status"], rec["profile"]["error"] = status, msg[-2000:]
        lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
        errors = [ln for ln in lines if "error" in ln.lower()]
        print(f"   {status}: {(errors or lines or [''])[0][:200]}")
        return rec

    env_build = dict(os.environ)
    if a.cxx:
        env_build["CXX"] = a.cxx
    if a.cxxflags is not None:
        env_build["CXXFLAGS"] = a.cxxflags
    try:
        b = subprocess.run([shell, "./build.sh"], cwd=scratch, env=env_build,
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return fail("build_failed", "build timed out")
    if b.returncode != 0 or not (scratch / "build" / "task").is_file():
        return fail("build_failed", b.stdout + b.stderr)
    built = [ln for ln in b.stdout.splitlines() if ln.startswith("built ")]
    ctx["builds"][name] = {"line": built[-1] if built else "",
                           "openmp": links_openmp(scratch / "build" / "task"),
                           "hpcbench_openmp": links_openmp(src / "build" / "task")}
    if name == ctx["baseline"] and built:
        print(f"   {built[-1]}", flush=True)

    work = scratch / "work"
    work.mkdir()
    task = str(scratch / "build" / "task")
    p = run_once([task, "prepare", str(ctx["raw"]), str(work)], scratch, ctx["env"], 900)
    if p["code"] != 0:
        return fail("prepare_failed", p["err"] or f"exit {p['code']}")

    tol, sig, threads = ctx["tol"], ctx["sig"], ctx["threads"]
    last = {}
    for k in ctx["ks"]:
        res, first = work / f"result_{k}.bin", work / f"first_{k}.bin"
        cmd = [task, "solve", str(work), str(res), str(k)]
        for _ in range(a.warmup):
            r = run_once(cmd, scratch, ctx["env"], ctx["timeout"])
            if r["code"] != 0:
                return fail("run_failed", f"k={k} warmup: " + (r["err"] or f"exit {r['code']}"))
        samples, effs, user, system, stable = [], [], 0.0, 0.0, True
        for i in range(a.runs):
            r = run_once(cmd, scratch, ctx["env"], ctx["timeout"])
            if r["code"] != 0:
                return fail("run_failed", f"k={k}: " + (r["err"] or f"exit {r['code']}"))
            t = timed_ms(r["out"])
            if t is None:
                return fail("run_failed", f"k={k}: no HPCBENCH_TIME_MS line on stdout")
            samples.append(round(t, 4))
            effs.append((r["user"] + r["sys"]) * 1000.0 / (t * threads) if t > 0 else 0.0)
            user, system = user + r["user"], system + r["sys"]
            rec["peak_rss_mb"] = max(rec["peak_rss_mb"], r["rss_mb"])
            if i == 0:
                shutil.copyfile(res, first)
            elif stable and not filecmp.cmp(res, first, shallow=False):
                stable = False
        try:
            digest, values = canon_digest(first, sig)
        except ValueError as e:
            return fail("wrong_answer", f"k={k}: {e}")
        med = statistics.median(samples)
        note = ""
        ref = ctx["ref_digests"].get(str(k))
        if name == ctx["baseline"]:
            ctx["base_values"][k] = values
        if digest != ref:
            base = ctx["base_values"].get(k)
            if base is None or name == ctx["baseline"]:
                note = "  differs from the reference digest"
            else:
                err = relative_error(values, base)
                ok = err < tol
                note = f"  differs from reference; rel. error {err:.1e} ({'within' if ok else 'OUTSIDE'} tolerance)"
                if not ok:
                    rec["status"] = "wrong_answer"
                    rec["profile"]["error"] = rec["profile"]["error"] or (
                        f"k={k}: relative error {err:.3e} exceeds the tolerance {tol}")
        if not stable:
            note += "  (results differ between runs)"
        print(f"   k={k}: {med:.2f} ms  {digest[:12]}…{note}", flush=True)
        rec["per_sweep"].append({"key": str(k), "median_ms": med, "digest": digest,
                                 "samples_ms": samples})
        last = {"effs": effs, "user": user, "sys": system}
    rec["cpu_efficiency"] = round(statistics.median(last["effs"]), 4) if last.get("effs") else 0.0
    rec["profile"]["user_s"] = round(last.get("user", 0.0), 3)
    rec["profile"]["sys_s"] = round(last.get("sys", 0.0), 3)
    rec["peak_rss_mb"] = round(rec["peak_rss_mb"], 2)
    return rec


# ---- main -------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Measure hpcbench entries locally; post nothing.")
    ap.add_argument("--workdir", default=".", help="the folder hpcbench pull wrote (default: .)")
    ap.add_argument("--only", nargs="*", help="measure only these entries (baseline is always included)")
    ap.add_argument("--runs", type=int, default=5, help="timed repetitions per k (default 5)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup repetitions per k (default 1)")
    ap.add_argument("--threads", type=int, default=None, help="default: the task's thread count")
    ap.add_argument("--ks", default=None, help="comma-separated sweep override, e.g. 8,64")
    ap.add_argument("--label", default=None, help="machine label (default: from an hpcbench bundle, else 'local')")
    ap.add_argument("--kind", default=None, help="machine kind (default: from an hpcbench bundle, else 'laptop')")
    ap.add_argument("--cxx", default=None, help="compiler passed to build.sh as CXX")
    ap.add_argument("--cxxflags", default=None, help="passed to build.sh as CXXFLAGS (default: build.sh's own)")
    ap.add_argument("--keep-build", action="store_true", help="keep .localbench/ (builds, prepared files)")
    a = ap.parse_args()

    wd = Path(a.workdir).resolve()
    manifest = json.loads((wd / "manifest.json").read_text())
    cfg = manifest.get("config", {})
    threads = a.threads or int(cfg.get("threads", 1))
    ks = [int(x) for x in a.ks.split(",")] if a.ks else [int(x) for x in cfg.get("sweep_values", [])]

    # Defaults for owner/label/kind/fingerprint from the newest hpcbench bundle, if any.
    prior = {}
    for path in sorted(wd.glob("bundle-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            b = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if b.get("format") == "hpcbench/device-run/1" and b.get("machine", {}).get("collected_by") == "hpcbench-runner":
            prior = b
            break
    label = a.label or prior.get("device", {}).get("label", "local")
    kind = a.kind or prior.get("device", {}).get("kind", "laptop")
    owner = prior.get("owner") or manifest.get("requested_by") or os.environ.get("USER", "")

    def is_entry(d):
        return all((d / f).is_file() for f in ("build.sh", "harness.cpp", "src/prepare.cpp", "src/solution.cpp"))

    folders = sorted(d.name for d in wd.iterdir() if d.is_dir() and not d.name.startswith(".") and is_entry(d))
    if "baseline" not in folders:
        sys.exit("no baseline/ entry in the workdir: hpcbench scores everything against it")
    others = [n for n in folders if n != "baseline"]
    if a.only is not None:
        missing = [n for n in a.only if n not in folders]
        if missing:
            sys.exit(f"not found in {wd}: {', '.join(missing)}")
        others = [n for n in others if n in a.only]
    entries = ["baseline"] + others

    def say(key, val):
        print(f"{key:<18}{val}", flush=True)

    cxx = a.cxx or os.environ.get("CXX", "c++")
    dev = device_info(cxx)
    say("task", f"{manifest.get('task')} — {manifest.get('title', '')} ({manifest.get('status', '')})")
    say("reported by", owner)
    say("sweep", f"{cfg.get('sweep_arg', 'k')} = " + ", ".join(str(k) for k in ks))
    say("entries", f"{len(entries)} to measure (local folders)")
    say("repetitions", f"{a.runs} timed, {a.warmup} warmup, {threads} threads")
    raw, input_info = ensure_input(wd, manifest, say)
    say("input", f"{raw} ({raw.stat().st_size / 1e6:.1f} MB)")
    say("machine", f"{dev['cpu']} · {dev['cores_logical']} cores · {dev['compiler']}")

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    root = wd / ".localbench"
    root.mkdir(exist_ok=True)
    ctx = {"wd": wd, "root": root, "raw": raw, "env": env, "ks": ks, "threads": threads,
           "sig": int(cfg.get("canon_sig_digits", 9)), "tol": float(cfg.get("correctness_rel_tol", 0.0)),
           "timeout": int(cfg.get("sanity_timeout_s", 120)), "baseline": "baseline", "base_values": {},
           "builds": {}, "ref_digests": {d["sweep_key"]: d["digest"] for d in manifest.get("reference_digests", [])},
           "reason": lambda n: "baseline — required for same-session scoring" if n == "baseline" else "local run"}

    results = []
    try:
        for name in entries:
            print(f"\n== {name}" + (" (baseline)" if name == "baseline" else "") + f" — {ctx['reason'](name)}", flush=True)
            results.append(measure_entry(name, a, ctx))
    finally:
        if not a.keep_build:
            shutil.rmtree(root, ignore_errors=True)

    bundle_id = uuid.uuid4().hex
    base_flags = ctx["builds"].get("baseline", {}).get("line", "")
    bundle = {
        "format": "hpcbench/device-run/1",
        "task": manifest.get("task"),
        "owner": owner,
        "device": {"fingerprint_id": prior.get("device", {}).get("fingerprint_id")
                                     or hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest()[:12],
                   "label": label, "kind": kind, **dev, "isa": []},
        "session": {"tier": "device", "runs": a.runs, "warmup": a.warmup,
                    "runner_version": "localbench-1", "bundle_id": bundle_id,
                    "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
        "machine": {"caches": None, "cpu_governor": None, "threads_requested": threads,
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
                    "omp_places": os.environ.get("OMP_PLACES"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
                    "python": platform.python_version(), "compiler": dev["compiler"],
                    "cxxflags": a.cxxflags if a.cxxflags is not None else base_flags.split(" ", 4)[-1] if base_flags else None,
                    "perf_available": shutil.which("perf") is not None, "collected_by": "localbench",
                    "input": input_info, "reduced_repetitions": a.runs < 5},
        "results": results,
    }
    out = wd / f"bundle-local-{bundle_id[:8]}.json"
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    print()
    # Did these builds use the same threading back-end as hpcbench's builds in the entry folders?
    omp = {n: b for n, b in ctx["builds"].items() if b["openmp"] is not None}
    linked = sorted(n for n, b in omp.items() if b["openmp"])
    differ = sorted(n for n, b in omp.items() if b["hpcbench_openmp"] is not None and b["openmp"] != b["hpcbench_openmp"])
    compared = sum(1 for b in omp.values() if b["hpcbench_openmp"] is not None)
    say("openmp", f"linked in {len(linked)} of {len(omp)} builds"
        + (f"; {compared} compared with hpcbench's builds, {len(differ)} differ" if compared else ""))
    for n in differ:
        yn = lambda v: "yes" if v else "no"
        print(f"{'':<18}{n}: this build {yn(omp[n]['openmp'])}, hpcbench's {yn(omp[n]['hpcbench_openmp'])}"
              " -- rerun with --cxxflags to match")
    say("bundle", str(out))
    print("local run — nothing was posted.")

    # Summary: speedup over this session's baseline.
    base = {s["key"]: s["median_ms"] for s in results[0]["per_sweep"]} if results and results[0]["status"] == "ok" else {}
    if base:
        keys = [str(k) for k in ks]
        print(f"\n{'speedup vs baseline':<20}" + "".join(f"{'k=' + k:>8}" for k in keys) + f"{'geomean':>10}  status")
        for r in results[1:]:
            med = {s["key"]: s["median_ms"] for s in r["per_sweep"]}
            if r["status"] == "ok" and all(k in med for k in keys):
                sp = [base[k] / med[k] for k in keys]
                gm = math.exp(sum(map(math.log, sp)) / len(sp))
                print(f"{r['submission']:<20}" + "".join(f"{x:8.1f}" for x in sp) + f"{gm:9.2f}x  ok")
            else:
                print(f"{r['submission']:<20}" + " " * 8 * len(keys) + f"{'—':>10}  {r['status']}")


if __name__ == "__main__":
    main()