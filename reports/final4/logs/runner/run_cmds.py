"""Command runner for the W17 confirmatory run (same record layout as reports/final2/logs/commands.json).

    python run_cmds.py --cwd C:\\dev_ML-wt-w17 --out-dir <folder> --python <python.exe> --commands commands.json [--stop-on-failure]

commands.json: [{"id": "1", "args": ["-m", "nbr", "tune", ...]}, ...]. Every command runs once, in order, with
NBR_DATA_DIR and PYTHONPATH=src set; stdout+stderr go to <out-dir>/cmd<id>.log (header lines with UTC start, cwd,
HEAD, dirty flag and the command). The process tree of each command is sampled every 0.5 s (Windows ToolHelp
snapshot + GetProcessMemoryInfo): peak_working_set_mib = largest per-process PeakWorkingSetSize seen,
peak_tree_working_set_sum_mib = largest sum of the current WorkingSetSize over the tree. Records are appended to
<out-dir>/commands.json after every command. With --stop-on-failure the chain stops at the first non-zero exit.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TH32CS_SNAPPROCESS = 0x00000002
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wt.DWORD), ("cntUsage", wt.DWORD), ("th32ProcessID", wt.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)), ("th32ModuleID", wt.DWORD), ("cntThreads", wt.DWORD),
                ("th32ParentProcessID", wt.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wt.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                             "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]


kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)
kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
kernel32.OpenProcess.restype = ctypes.c_void_p
kernel32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wt.DWORD]


def parent_map() -> dict[int, int]:
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == INVALID_HANDLE_VALUE:
        return {}
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    parents = {}
    try:
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return parents


def tree_pids(root: int) -> list[int]:
    parents = parent_map()
    pids, frontier = [root], [root]
    while frontier:
        nxt = [pid for pid, parent in parents.items() if parent in frontier and pid not in pids]
        pids += nxt
        frontier = nxt
    return pids


def memory(pid: int) -> tuple[float, float] | None:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return None
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / 2**20, counters.WorkingSetSize / 2**20
    finally:
        kernel32.CloseHandle(handle)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def run_one(item: dict, cwd: Path, out_dir: Path, python: str, env: dict) -> dict:
    log_path = out_dir / f"cmd{item['id']}.log"
    commit = git(cwd, "rev-parse", "HEAD")
    dirty = bool(git(cwd, "status", "--porcelain"))
    argv = [python, *item["args"]]
    start = utc()
    t0 = time.perf_counter()
    peak_single, peak_sum, samples, max_procs = 0.0, 0.0, 0, 0
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        log.write(f"# {start} cwd={cwd.as_posix()} commit={commit} dirty={dirty}\n")
        log.write("# " + " ".join(argv) + "\n")
        log.flush()
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            exited = proc.poll()
            pids = tree_pids(proc.pid)
            total = 0.0
            for pid in pids:
                mem = memory(pid)
                if mem:
                    peak_single = max(peak_single, mem[0])
                    total += mem[1]
            peak_sum = max(peak_sum, total)
            samples += 1
            max_procs = max(max_procs, len(pids))
            if exited is not None:
                break
            time.sleep(0.5)
    end = utc()
    return {"id": item["id"], "command": "python " + " ".join(item["args"]), "commit": commit, "worktree_dirty": dirty,
            "start_utc": start, "end_utc": end, "wall_seconds": round(time.perf_counter() - t0, 1), "exit_code": proc.returncode,
            "peak_working_set_mib": round(peak_single, 1), "peak_tree_working_set_sum_mib": round(peak_sum, 1),
            "memory_samples": samples, "max_processes_in_tree": max_procs, "log": f"logs/{log_path.name}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--commands", required=True)
    parser.add_argument("--data-dir", default=r"C:\dev_ML\data\processed")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args()
    cwd, out_dir = Path(args.cwd), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "NBR_DATA_DIR": args.data_dir, "PYTHONPATH": "src"}
    commands = json.loads(Path(args.commands).read_text(encoding="utf-8"))
    records_path = out_dir / "commands.json"
    records = json.loads(records_path.read_text(encoding="utf-8")) if records_path.exists() else []
    for item in commands:
        record = run_one(item, cwd, out_dir, args.python, env)
        records.append(record)
        records_path.write_text(json.dumps(records, indent=2), encoding="utf-8", newline="\n")
        print(json.dumps(record))
        sys.stdout.flush()
        if record["exit_code"] != 0 and args.stop_on_failure:
            print(f"command {item['id']} failed with exit code {record['exit_code']}; chain stopped")
            sys.exit(record["exit_code"])


if __name__ == "__main__":
    main()
