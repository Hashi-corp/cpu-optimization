from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
OUTPUTS_ROOT = ROOT / "outputs"
VENV_ACTIVATE_PS1 = ROOT / ".venv" / "Scripts" / "Activate.ps1"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


def ensure_project_venv() -> None:
    """Re-launch this script with project .venv Python when available."""
    if os.getenv("CPU_OPT_VENV_BOOTSTRAPPED") == "1":
        return

    if not VENV_PYTHON.exists():
        return

    current_python = Path(sys.executable).resolve()
    target_python = VENV_PYTHON.resolve()
    if current_python == target_python:
        return

    env = dict(os.environ)
    env["CPU_OPT_VENV_BOOTSTRAPPED"] = "1"
    command = [str(target_python), str(Path(__file__).resolve()), *sys.argv[1:]]
    print("Using project virtual environment automatically:")
    print(f"  {target_python}")
    result = subprocess.run(command, cwd=str(ROOT), env=env)
    raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all benchmark outputs in one command and store each run in outputs/outputs(N).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full native benchmark path (slower). By default fast demo mode is used.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=90,
        help="Timeout per script execution.",
    )
    return parser.parse_args()


def next_run_dir(outputs_root: Path) -> Path:
    outputs_root.mkdir(parents=True, exist_ok=True)
    existing = []
    for child in outputs_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("outputs(") and name.endswith(")"):
            number = name[len("outputs(") : -1]
            if number.isdigit():
                existing.append(int(number))
    run_id = (max(existing) + 1) if existing else 1
    run_dir = outputs_root / f"outputs({run_id})"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_script(script_name: str, fast_demo: bool, timeout_seconds: int) -> None:
    target_script = ROOT / script_name
    env = dict(**os.environ)
    if fast_demo:
        env["CPU_OPT_FAST_DEMO"] = "1"

    if VENV_ACTIVATE_PS1.exists():
        ps_command = (
            f"& '{VENV_ACTIVATE_PS1}'; "
            f"python '{target_script}'"
        )
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_command,
        ]
        print(f"Running with venv activation: {target_script.name}")
    elif VENV_PYTHON.exists():
        command = [str(VENV_PYTHON), str(target_script)]
        print(f"Running with venv python: {target_script.name}")
    else:
        command = [sys.executable, str(target_script)]
        print(f"Running with current python: {target_script.name}")

    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {proc.returncode}")


def collect_generated_artifacts(start_ts: float) -> List[Path]:
    patterns = [
        "semantic_cache_*.csv",
        "semantic_cache_*.png",
        "feature_benchmark_*.csv",
        "feature_benchmark_*.png",
    ]
    files: List[Path] = []
    for pattern in patterns:
        for path in ROOT.glob(pattern):
            if path.is_file() and path.stat().st_mtime >= (start_ts - 1):
                files.append(path)
    return sorted(set(files))


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def best_row_by_float(rows: List[Dict[str, str]], key: str) -> Dict[str, str]:
    if not rows:
        return {}
    return min(rows, key=lambda row: float(row.get(key, "inf") or "inf"))


def build_final_summary(run_dir: Path, mode: str, run_seconds: float) -> Path:
    semantic_rows = read_rows(ROOT / "semantic_cache_summary.csv")
    feature_rows = read_rows(ROOT / "feature_benchmark_summary.csv")

    semantic_best = best_row_by_float(semantic_rows, "avg_latency_ms")
    feature_best = best_row_by_float(feature_rows, "latency_ms")

    summary_path = run_dir / "final_demo_summary.txt"
    lines = [
        "CPU Optimization Demo Final Summary",
        "===================================",
        f"Generated at (UTC): {datetime.now(UTC).isoformat()}",
        f"Run mode: {mode}",
        f"Total runtime (seconds): {run_seconds:.2f}",
        "",
        "Semantic Cache Benchmark:",
    ]

    if semantic_best:
        lines.extend(
            [
                f"- Best strategy by avg latency: {semantic_best.get('strategy', 'n/a')}",
                f"- Avg latency (ms): {semantic_best.get('avg_latency_ms', 'n/a')}",
                f"- Speedup vs baseline: {semantic_best.get('speedup_vs_baseline', 'n/a')}x",
                f"- Hit rate (%): {semantic_best.get('hit_rate_pct', 'n/a')}",
            ]
        )
    else:
        lines.append("- No semantic summary rows found")

    lines.append("")
    lines.append("Feature Benchmark (#3-#9):")
    if feature_best:
        lines.extend(
            [
                f"- Best strategy by latency: {feature_best.get('strategy', 'n/a')}",
                f"- Issue: {feature_best.get('issue', 'n/a')}",
                f"- Latency (ms): {feature_best.get('latency_ms', 'n/a')}",
                f"- Throughput (qps): {feature_best.get('throughput_qps', 'n/a')}",
            ]
        )
    else:
        lines.append("- No feature summary rows found")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def _extract_issue_number(issue_label: str) -> str:
    match = re.search(r"#(\d+)", issue_label or "")
    return match.group(1) if match else "unknown"


def generate_per_issue_feature_artifacts(run_dir: Path) -> List[str]:
    feature_csv = ROOT / "feature_benchmark_summary.csv"
    if not feature_csv.exists():
        return []

    rows = read_rows(feature_csv)
    if not rows:
        return []

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        issue_number = _extract_issue_number(row.get("issue", ""))
        grouped.setdefault(issue_number, []).append(row)

    created_files: List[str] = []
    headers = list(rows[0].keys())

    for issue_number, issue_rows in grouped.items():
        issue_csv_name = f"feature_benchmark_summary_issue_{issue_number}.csv"
        issue_csv_path = run_dir / issue_csv_name
        with issue_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(issue_rows)
        created_files.append(issue_csv_name)

        # Create a light per-issue dashboard from the combined summary rows.
        issue_png_name = f"feature_benchmark_dashboard_issue_{issue_number}.png"
        issue_png_path = run_dir / issue_png_name
        try:
            import matplotlib.pyplot as plt

            strategies = [row.get("strategy", "n/a") for row in issue_rows]
            latencies = [float(row.get("latency_ms", "0") or "0") for row in issue_rows]
            sorted_pairs = sorted(zip(strategies, latencies), key=lambda item: item[1])
            sorted_strategies = [item[0] for item in sorted_pairs]
            sorted_latencies = [item[1] for item in sorted_pairs]

            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            ax.barh(sorted_strategies, sorted_latencies, color="#2E86AB")
            ax.set_xlabel("Latency (ms)")
            ax.set_title(f"Feature Benchmark Issue #{issue_number} — Latency")
            plt.tight_layout()
            plt.savefig(issue_png_path, bbox_inches="tight")
            plt.close(fig)
            created_files.append(issue_png_name)
        except Exception:
            # Keep the run robust even if plotting is unavailable.
            pass

    return created_files


def main() -> None:
    ensure_project_venv()

    args = parse_args()
    fast_demo = not args.full
    mode = "fast-demo" if fast_demo else "full"

    run_dir = next_run_dir(OUTPUTS_ROOT)
    started = time.time()

    run_script("semantic_cache_extension.py", fast_demo=fast_demo, timeout_seconds=args.timeout_seconds)
    run_script("cpu_optimization_feature_benchmark.py", fast_demo=fast_demo, timeout_seconds=args.timeout_seconds)

    artifacts = collect_generated_artifacts(started)
    for artifact in artifacts:
        shutil.copy2(artifact, run_dir / artifact.name)

    per_issue_artifacts = generate_per_issue_feature_artifacts(run_dir)

    elapsed = time.time() - started
    summary_path = build_final_summary(run_dir, mode=mode, run_seconds=elapsed)

    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "runtime_seconds": round(elapsed, 3),
        "artifact_count": len(artifacts) + len(per_issue_artifacts),
        "artifacts": [path.name for path in artifacts] + per_issue_artifacts,
        "final_summary": summary_path.name,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("All outputs generated successfully.")
    print(f"Run directory      : {run_dir}")
    print(f"Final summary file : {summary_path.name}")
    print(f"Artifacts copied   : {len(artifacts)}")
    print(f"Per-issue artifacts: {len(per_issue_artifacts)}")
    print(f"Runtime (seconds)  : {elapsed:.2f}")


if __name__ == "__main__":
    main()
