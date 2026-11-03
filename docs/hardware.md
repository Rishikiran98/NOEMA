# Hardware detection and the CPU-only launcher (v0.5.2)

Nothing is tuned for a particular machine any more. `cpu_platform.hardware` inspects
the computer it runs on, and `cpu_platform.launch` starts llama.cpp from that profile,
the same way on Windows, Linux and macOS.

```bash
python -m cpu_platform.hardware              # what was detected + recommended settings
python -m cpu_platform.launch --list-models  # presets, whether each fits in RAM, speed estimate
python -m cpu_platform.launch --model qwen3-0.6b               # serve on 127.0.0.1:8080
python -m cpu_platform.launch --model qwen3-4b --port 8081     # second model for experiment 6
python -m cpu_platform.launch --model qwen3-4b --dry-run       # print command + manifest only
```

`scripts/serve_windows_cpu.ps1` and `scripts/serve_cpu.sh` are thin wrappers that pass
their arguments to the launcher.

## What is detected

| Item | How | Used for |
| --- | --- | --- |
| Physical cores | `/proc/cpuinfo`/`lscpu`, `sysctl hw.physicalcpu`, `Win32_Processor.NumberOfCores` | `--threads`, `--threads-batch` (hyperthreads usually slow token generation) |
| RAM | `/proc/meminfo`, `sysctl hw.memsize`, `GlobalMemoryStatusEx` | context size; which models fit (weights + headroom ≤ 60% of RAM) |
| GPUs | `nvidia-smi`, `Win32_VideoController`, `/sys/class/drm`, Apple Silicon | hiding every GPU, including integrated ones (Vulkan can use them) |
| Copy bandwidth | ~1 s single-thread copy of 256 MB | rough tokens/s estimate (bandwidth / model size) |

The tokens/s figure is an order-of-magnitude estimate, not a measurement. Generating a
token on a CPU streams all the weights through memory once, so bandwidth / model size is
what makes a small model so much cheaper per token. Experiment 6 is built on that.

## How "CPU-only" is enforced and recorded

1. `--n-gpu-layers 0` (no layers offloaded).
2. `--device none` when the build supports it. This stops prompt processing from quietly
   using a GPU in CUDA/Vulkan builds even with 0 offloaded layers.
3. Environment variables that hide devices from every backend: `CUDA_VISIBLE_DEVICES`,
   `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `ONEAPI_DEVICE_SELECTOR`, `GGML_VK_VISIBLE_DEVICES`.
4. The server log is saved to `results/llama_server_<port>.log` and scanned for GPU
   evidence ("found 1 CUDA devices", "using device CUDA0", "offloaded 12/29 layers to GPU").
   The verdict goes into `results/launch_<port>.json`:
   - `no_gpu_backend_seen`: a CPU-only build.
   - `gpu_backend_mentioned_not_used`: a GPU-capable build with no device visible or used. Acceptable, but say so when reporting.
   - `gpu_visible_or_used`: do **not** report the run as CPU-only. Use a CPU-only build (`scripts/build_cpu_llama.sh`).

This is evidence, not hardware attestation. Also watch GPU utilisation in your OS monitor
(Task Manager → Performance → GPU on Windows) during a run. `--allow-gpu` turns all of this
off, and the manifest then records `"requested": false`.

## Qwen3 thinking mode

Qwen3 models "think" before answering by default, and that can use up the whole token
budget before any JSON appears, so every mode would score near zero. Since v0.8 thinking is
decided per request: the launcher passes `--jinja --reasoning auto`, and the client sends
`enable_thinking` (off unless a request asks for it), appends `/no_think` when off, and strips
any `<think>` block.

Only **hybrid-thinking** models get these switches (M1.2). Qwen3-Coder and the Qwen3 2507
instruct/thinking releases are not hybrid, and neither is Qwen2.5-Coder, so they get no
`/no_think`, no `enable_thinking` and no `--reasoning`; Qwen3-family models still get `--jinja`.
Presets declare this (`thinking` in `--list-models`); for `--hf` / `--model-file`, the name
decides (`cpu_first.core.hybrid_thinking`).

## The launch manifest

`results/launch_<port>.json` records the exact argv, llama.cpp version, threads, context
size, model source, PID, the full hardware profile and the GPU log scan. Experiments 4 and
6 read it: `--launch-manifest` (experiment 4), or automatically by port (experiment 6),
which also uses the PID to measure the server's CPU-seconds when `psutil` is installed.

## Model presets

`qwen3-0.6b` (Q8_0, ~0.64 GB), `qwen3-1.7b` (Q8_0, ~1.8 GB), `qwen3-4b` (Q4_K_M, ~2.5 GB),
`qwen3-8b` (Q4_K_M, ~5.0 GB), `qwen3-14b` (Q4_K_M, ~9.0 GB). Sizes are approximate and the
upstream repositories can change, so use `--hf REPO:QUANT` or `--model-file x.gguf` for
anything else. The launcher refuses a model that won't fit in RAM unless you pass `--force`.

**Coder presets (NOEMA Engine, M1.2).** Both default to a 16,384-token context (override with `--ctx-size`).

| Preset | Source | Params | File | Needs at 16k ctx | Fits |
|---|---|---|---|---|---|
| `qwen3-coder-30b-a3b` | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_M` (third-party GGUF; Qwen publishes none) | 30.5B MoE, 3.3B active | ~18.6 GB | ~20.6 GB | 32 GB machines |
| `qwen2.5-coder-7b` | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M` | 7.6B dense | ~4.7 GB | ~6.1 GB | 16 GB machines |

The MoE model needs RAM for all 30.5B parameters but reads only the ~3.3B active ones per
token, so it should generate faster than its file size suggests; that is an estimate until
measured. "Needs" = file + f16 KV cache (per-model layer/head shape) + 0.5 GB buffers.

**RAM budget.** A model must fit in the larger of 60% of RAM and RAM minus 10 GB. The fixed
reserve reflects that the OS and API need a roughly constant amount; it lets the 30B coder
(~20.6 GB) run on a 32 GB laptop (~22 GB budget), with about 10 GB left for everything else.
Close other large programs before launching it.
