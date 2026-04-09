# CPU Optimization Project Revision Guide (Judge-Ready)

## 1) One-minute opening pitch

This project is a CPU-focused LLM optimization benchmark suite.

The goal is simple: reduce latency and improve throughput for text generation workloads on CPU-only systems, while measuring real trade-offs in memory and output quality.

I implemented and benchmarked multiple optimization strategies:

- semantic and exact-response caching,
- speculative decoding,
- thread tuning and concurrent sampling,
- profiling for CPU and memory,
- ONNX Runtime and OpenVINO inference paths,
- and generation diversity controls.

Then I built dashboards and summary tables to compare all strategies side by side.

## 2) Problem statement and why it matters

Many production environments cannot rely on GPUs because of cost, availability, or deployment constraints.

CPU optimization matters because:

- CPU instances are cheaper and more widely available.
- Enterprise and edge environments often require CPU-only deployments.
- Lower latency and better throughput directly improve user experience and infra cost.

So this project is about practical optimization techniques that can be measured and explained, not just theory.

## 3) Project architecture in plain language

The project has two benchmark tracks:

- Core semantic cache track
- Feature track for issues #3 to #9

### Core semantic cache track

`semantic_cache_extension.py` runs three strategies over the same prompt workload:

- Baseline (no cache)
- Exact cache
- Semantic cache

It generates:

- summary CSV,
- trace CSV,
- latency comparison plot,
- hit breakdown plot,
- query-sequence savings plot.

### Feature track (#3 to #9)

`cpu_optimization_feature_benchmark.py` benchmarks:

- #3 OpenVINO path,
- #4 speculative decoding,
- #5 multithreading,
- #6 CPU and memory profiling,
- #7 output diversity controls,
- #8 ONNX Runtime integration,
- #9 concurrent sampling.

It generates:

- summary CSV,
- trace CSV,
- dashboard PNG.

## 4) Key files you should know before judging

- `cpu_utils.py`: CPU thread and profiling utility helpers.
- `kv_cache.py`: cache logic and semantic cache support.
- `onnx_optimizer.py`: ONNX export, quantization, and ORT inference wrapper.
- `semantic_cache_extension.py`: benchmark for baseline vs exact vs semantic cache.
- `cpu_optimization_feature_benchmark.py`: issue-based feature benchmark runner.
- `run_all_outputs_demo.py`: one-command script to run both benchmark tracks and archive outputs.

## 5) What I changed for live demo reliability

I added a demo-safe flow so results can be generated quickly and repeatedly in front of judges.

### Fast demo mode

When environment variable `CPU_OPT_FAST_DEMO=1` is active:

- benchmark scripts force simulated backend for speed and robustness,
- heavy model operations are reduced,
- OpenVINO and ONNX sections use fast approximated metrics,
- workload size is trimmed.

This keeps the run fast and predictable for live evaluation.

### One-command run and run folders

`run_all_outputs_demo.py` now:

- runs semantic benchmark and feature benchmark in sequence,
- creates `outputs/outputs(1)`, `outputs/outputs(2)`, ... automatically,
- copies all new artifacts into that run folder,
- writes a final summary text file,
- writes a run manifest JSON for traceability.

## 6) Exactly how to run during judging

From the `cpu-optimization` folder:

```powershell
.\.venv\Scripts\python .\run_all_outputs_demo.py
```

This uses fast demo mode by default.

If someone asks for a native/full run:

```powershell
.\.venv\Scripts\python .\run_all_outputs_demo.py --full
```

After each run, open the newest folder in `outputs`.

Inside it, highlight:

- `final_demo_summary.txt` (single final output),
- `run_manifest.json` (what was generated),
- all CSV and PNG artifacts for evidence.

## 7) How to explain the benchmark methodology

Use this structure:

1. Same workload across strategies for fairness.
2. Measured latency, throughput, memory, and cache behavior.
3. Used summary + trace outputs for both aggregate and per-query analysis.
4. Visualized results in dashboards so trends are easy to compare.

If asked about semantic cache quality:

- exact cache is strict key match,
- semantic cache supports near-duplicate/paraphrased prompts,
- this increases hit opportunities and can reduce average latency.

## 8) Expected judge questions and strong answers

### Q1: Why both exact and semantic cache?

Exact cache is fast but only handles identical prompts.
Semantic cache captures paraphrases and near-duplicates, so practical hit rate is higher in realistic workloads.

### Q2: Why include ONNX and OpenVINO?

They represent important CPU inference deployment paths.
ONNX Runtime gives graph-level and runtime optimizations; OpenVINO targets Intel CPU acceleration.

### Q3: What is the trade-off with diversity settings?

Higher diversity often increases response variety but may increase latency slightly.
The benchmark reports both quality proxy (`distinct1`) and performance metrics.

### Q4: How do you ensure repeatability?

- fixed scripted workload,
- standardized output files,
- per-run archived output folders,
- final summary and run manifest for every execution.

### Q5: What should be productionized next?

- add automatic regression thresholds,
- pin environment versions and model hashes,
- add CI pipeline that runs reduced benchmark and verifies metric bounds.

## 9) 2-minute demo talk track

1. Start with problem statement: CPU-only inference optimization.
2. Run one command: `run_all_outputs_demo.py`.
3. Show generated run folder (`outputs(N)`).
4. Open `final_demo_summary.txt` first (single judge-friendly output).
5. Open dashboard PNGs for visual proof.
6. Open CSVs briefly to show raw benchmark data.
7. Close with key improvement narrative and reproducibility.

## 10) Personal checklist before presenting

- Confirm virtual environment is active.
- Run one dry run before judges arrive.
- Keep `outputs` folder visible in explorer.
- Know where `final_demo_summary.txt` is.
- Practice the 1-minute pitch and 2-minute talk track.

You are ready if you can explain:

- what was optimized,
- how it was measured,
- what improved,
- and how results are reproducible in one command.
