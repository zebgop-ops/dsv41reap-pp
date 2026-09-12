# Findings

What it took to serve the REAP-272E checkpoint on four sm_80 cards with every expert on the GPUs, and what limits it. The debugging story of the shared overlay (Ampere attention path, Engram on NVMe, PP shadow sources, speculative decoding) is in [dsv41flash-pp/FINDINGS.md](https://github.com/zebgop-ops/dsv41flash-pp/blob/main/FINDINGS.md).

## 1. Every expert on the GPUs

`LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E` keeps 272 of the 384 routed experts per layer
(router-weighted pruning; the card reports +4.1% text perplexity), same dense weights, same
Engram tables (not in the repo: link the base model's shards 47/48 in, *relatively*, because the
cache is bind-mounted at `/hf` inside the container; `link-reap-engram.sh`). Experts drop from
6.72 to 4.76 GiB per layer, which changes the memory arithmetic that shaped the unpruned port (dsv41flash-pp).

- **Partition.** Legal PP cuts sit on KV/index-source layers (2, 8, 14, 20, 24, 28, 32, 36) and
  no 4-way split of those keeps every rank at 11 expert layers or fewer, so 8,12,8,12 would still
  need four layers on the CPU. The shadow plan turns out to handle cuts *inside* an index group:
  the later rank gets a shadow of the group's index source that replays its top-k (the helper
  that forbids such cuts was never wired in). With 10,10,10,10 rank 1 shadows source 8 and rank 3
  shadows 20 and 28, and all 40 layers' experts fit (~52 GiB per rank, KV pool 4.7M tokens).
  Validated against 8,12,8,12: same probe texts, identical greedy code-edit output, prose
  diverging at char 429 (the stack's usual kernel numerics).
- **Router.** vLLM's Triton DSv4 top-k kernel takes any expert count (it pads to a power of two
  and masks) but its admission check only allows 256 or 384, so 272 fell through to a CUDA
  kernel with a fixed expert table: `Unsupported expert number: 272`, the reason the card says
  the checkpoint "cannot run on vLLM". `patch_reap_router.py` relaxes the check.
- **Result.** With no CPU experts the decode step is ~37 ms and prefill runs at ~500 tok/s;
  speculative verify rows are cheap again, so DSpark (default here, on a 10,10,11,9 partition so
  its draft fits next to 9 layers on rank 3) reaches ~35 tok/s on prose and ~100 tok/s decode on
  code, and n-gram ~66 tok/s on copy-heavy code. Numbers in RESULTS.md.
- **Prefill.** With a drafter on, the engine core blocks in `take_draft_token_ids` after every
  batch, so a long prompt's 2048-token chunks went through the four ranks strictly one at a
  time: 1.4k tok/s, flat from 6k to 21k tokens, ~1.47 s per chunk = the sum of the four ranks.
  The overlay's `core.py` now skips that RPC for batches that do not complete a prompt
  (`DSV41_DRAFT_SKIP_PREFILL=1`), which lets the batch queue keep up to four chunks in flight:
  2.4k tok/s at 6k and 3.1k tok/s at 14k tokens, outputs unchanged. What remains is the per-rank
  chunk compute (~650 ms per 2048 tokens on the slowest rank) and the queue depth.

## 2. Long context, safe memory margins, and where speed comes from

- **Context.** `--max-model-len 524288` with utilization 0.93: KV pool ~3M tokens (about six
  512k requests), needle recall verified at 30k and 200k tokens. A 1M limit boots only with every
  rank at 10 layers and nothing else on them: the profiling transient at 1M plus an 11-layer rank
  (55 GiB of weights) or the DSpark draft leaves no memory for KV blocks.
- **Memory margins.** Cards on this box have crashed when driven to 63 GiB of 64, so the launcher
  runs at 0.93 (~3.5 GiB of slack per card). The partition has to be 10,10,10,10 for that: an
  11-layer rank at 0.95 and 256k context had 0.08 GiB left for KV.
- **Prefill** is compute-bound in the Marlin MXFP4 expert GEMMs: ~400-500 ms per 2048-token
  chunk per rank (~40 ms per layer, ~1.7 TFLOP of expert GEMM per layer), flat from 6k to 21k
  tokens, and the four ranks overlap consecutive chunks. One fix mattered: with a drafter on, the
  engine core blocked on the draft-token RPC after every batch, which serialized prefill chunks
  through the pipeline (1.4k tok/s, ~1.47 s per chunk = the sum of the ranks). Skipping that RPC
  for batches that do not complete a prompt (`DSV41_DRAFT_SKIP_PREFILL=1` in the overlay's
  `core.py`) lets the batch queue keep four chunks in flight: 2.4k tok/s at 6k, 3.1k at 14k.
- **Decode** is ~29-37 ms of GPU time per step across the four ranks plus ~5-7 ms of engine round
  trip and rank-0 host prep that only async scheduling could hide (vLLM disables it for CPU n-gram
  and for DSpark under PP). That is within ~15% of the hardware's ceiling with PP4 over PCIe Gen2.
- **DSpark** (`DSV41_SPEC_METHOD=dspark`, the V1-runner port from dsv41flash-pp) doubles fresh-code
  speed (300 tokens in 3.4 s vs ~11 s) but its 8.4 GiB draft only fits on the last rank with that
  rank's last two layers' experts on the CPU; the default keeps every layer on the GPUs.

## Diagnostic switches (all off by default)

| switch | effect |
|---|---|
| `DSV41_CG=NONE` | no CUDA graphs (eager), for tracebacks that point at the right kernel |
| `DSV41_EXTRA_DOCKER="-e CUDA_LAUNCH_BLOCKING=1 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=0"` | synchronous launches; the exception is logged before the NCCL watchdog aborts |
| `DSV41_DEBUG_STATS=1` | per-layer \|h\| / residual stats every step (each line syncs) |
| `DSV41_DEBUG_DUMP=/dump`, `DSV41_DEBUG_DUMP_LAYERS=0,1,2` | attn/ffn input+output dumps for 1 < T ≤ 64 batches, plus per-stage attention dumps for `ktests/ref_stages.py` |
| `DSV41_DEBUG_SYNC=1` | device syncs around the PP receive and per KV group in slot mapping; logs block-table geometry and per-step recv-wait/forward times |
| `DSV41_DEBUG_TIMING=1` | per-layer wall time (with syncs) every 16th decode step |
| `DSV41_DEBUG_PROFILE=1` | wraps the 41st decode step in `torch.profiler` and logs the top kernels by device time per rank |
| `DSV41_DEBUG_SPEC=1` | logs every small batch's CUDA-graph dispatch (tokens, requests, uniform, mode, descriptor) |
| `DSV41_DEBUG_ENGRAM=1` | checks the static Engram metadata buffers against the live attention metadata every step |
| `DSV41_DEBUG_GDUMP=/dump` | graph-safe per-layer / sub-block activation dumps (`cudaLaunchHostFunc` + pinned buffers), compared with `ktests/diff_gdump.py` |
| `DSV41_DEBUG_CORE=1` | engine-core phase means (schedule, RPC issue, wait, take_draft, update) every 32 real steps |
| `DSV41_DEBUG_TRACE=1` | cross-rank timeline of decode steps 40-44 (worker entry, receive posted, launched, sent, GPU done, engine issue) rendered by `tools/tracetl.py` |
| `DSV41_FULL_Q1=0` | upstream behaviour under speculation: capture sizes rounded to K+1, draft-less steps in PIECEWISE graphs |
| `DSV41_PREFILL_EAGER=0` | let prefill-shaped batches use piecewise graphs |
| `DSV41_DEBUG_DSPARK=1` | per-phase timing of the DSpark draft step (with syncs), every 32 calls |
| `CPU_MOE_TRACE=1` | the native CPU kernel prints its phase times |
| `DSV41_ENGRAM_ZERO=1` | Engram lookups return zeros (isolates the SSD path) |
| `DSV41_MHC_TORCH=1` | torch mHC instead of tilelang |
