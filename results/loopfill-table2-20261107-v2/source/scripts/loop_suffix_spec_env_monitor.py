"""Host-side environment monitor for wall-clock benchmarks (standard library only).

Run on the host, outside the benchmark containers, pinned to a CPU the benchmark does
not use. Every --interval seconds it appends one JSON line: per-GPU utilization,
memory, SM/memory clocks, temperature, power and active clock-event (throttle) reasons;
every compute process on any GPU with its owner and command; busy fraction and current
frequency of the watched CPUs; and the load average. loop_suffix_spec_wallclock.py
--summarize --monitor joins it with the per-run timestamps.

  taskset -c 0 python3 scripts/loop_suffix_spec_env_monitor.py --cpus 16-23,64-71 --out m.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import subprocess
import time
from pathlib import Path

GPU_FIELDS = [
    "index", "uuid", "utilization.gpu", "memory.used", "clocks.sm", "clocks.mem",
    "temperature.gpu", "power.draw",
]  # fmt: skip
REASON_FIELDS = ["clocks_event_reasons.active", "clocks_throttle_reasons.active"]


def parse_cpus(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def smi(query: str, fields: list[str]) -> list[list[str]]:
    res = subprocess.run(
        ["nvidia-smi", f"--query-{query}={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [[x.strip() for x in line.split(",")] for line in res.stdout.splitlines() if line]


def num(x: str):
    try:
        return float(x) if "." in x else int(x, 0)
    except ValueError:
        return None


def cpu_times() -> dict[int, tuple[int, int]]:
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            name, *vals = line.split()
            v = [int(x) for x in vals]
            out[int(name[3:])] = (v[3] + v[4], sum(v[:8]))  # idle + iowait, total
    return out


def owner(pid: int) -> tuple[str | None, str | None]:
    """(user name, or the numeric uid when it has no passwd entry; command name)."""
    try:
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        uid = next(int(line.split()[1]) for line in status if line.startswith("Uid:"))
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
    except (OSError, StopIteration):
        return None, None
    try:
        return pwd.getpwuid(uid).pw_name, comm
    except KeyError:
        return str(uid), comm


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--cpus", default="", help="CPUs to report, e.g. 16-23,64-71")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    cpus = parse_cpus(args.cpus) if args.cpus else []
    reason_field = next(
        (f for f in REASON_FIELDS if subprocess.run(
            ["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"], capture_output=True
        ).returncode == 0),
        None,
    )  # fmt: skip
    fields = GPU_FIELDS + ([reason_field] if reason_field else [])
    prev = cpu_times()
    with open(args.out, "a") as fh:
        while True:
            time.sleep(args.interval)
            now = cpu_times()
            busy = {
                c: round(1 - (now[c][0] - prev[c][0]) / max(1, now[c][1] - prev[c][1]), 3)
                for c in cpus
                if c in now
            }
            prev = now
            gpus = []
            for row in smi("gpu", fields):
                g = dict(zip(["index", "uuid", "util", "mem_mib", "sm_mhz", "mem_mhz", "temp_c",
                              "power_w", "reasons"], row, strict=False))  # fmt: skip
                gpus.append({k: (v if k == "uuid" else num(v)) for k, v in g.items()})
            procs = []
            for uuid, pid, mem in smi("compute-apps", ["gpu_uuid", "pid", "used_memory"]):
                user, comm = owner(int(pid))
                procs.append(
                    {
                        "gpu_uuid": uuid,
                        "pid": int(pid),
                        "mem_mib": num(mem),
                        "user": user,
                        "comm": comm,
                    }
                )
            freq = {}
            for c in cpus:
                f = Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
                if f.exists():
                    freq[c] = int(f.read_text()) // 1000
            rec = {
                "time": time.time(),
                "loadavg": os.getloadavg(),
                "gpus": gpus,
                "procs": procs,
                "cpu_busy": busy,
                "cpu_mhz": freq,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
