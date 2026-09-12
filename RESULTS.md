# Results

DeepSeek-V4.1-Flash-REAP-272E on 4× CMP 170HX (sm_80, 64 GB, PCIe Gen2 x4), vLLM
`deepseekv41-flash-0909` + overlay, PP4 partition 10,10,10,10, every expert on the GPUs, n-gram
speculation (K=3, prompt-lookup 5..8), utilization 0.93, `--max-model-len 524288`. Single stream,
greedy, thinking off unless noted; "incl. prefill" numbers are wall time for the whole request.

## Correctness

- Greedy output of a 400-token code-edit task is byte-identical to the unpruned model's (and to the
  8,12,8,12 layout's, which only cuts on source boundaries).
- Needle-in-haystack recall OK at 30k and 200k tokens.
- Three logprob probes: same texts as the unpruned model, flatter distributions (up to 2.5-4.3 nats
  on the top token); the model card reports +4.1% text perplexity for the pruning.

## Speed

| workload | unpruned model (9 CPU layers, dsv41flash-pp) | REAP-272E, all GPU |
|---|---|---|
| free prose, 400 tokens streaming | 26-28 s (14 tok/s) | 12.7-14.2 s (28-31 tok/s, ~37 ms/step) |
| code edit, 529-token prompt, 400 tokens streaming | 23.5 s | 6.0 s (129 steps, ~66 tok/s decode) |
| specbench prose / code-edit (thinking on, incl. prefill) | 13.6 / 14.8 tok/s | 24-28 / 43-45 tok/s |
| prefill, 525-token prompt | ~60 tok/s | ~500 tok/s |
| prefill, 6k / 14k-token prompts | ~100 tok/s | 2.4k / 3.1k tok/s |
| prefill, 29k tokens (needle) | ~106 tok/s | 316 tok/s before the chunk-pipelining fix |
| KV pool | 0.9M tokens @131k | 2.2M tokens @512k at utilization 0.93 (3.7M at 0.95) |

With the model's own DSpark drafter (`DSV41_SPEC_METHOD=dspark`, two of rank 3's layers on the CPU):
prose 27-34 tok/s, code edit 4.3 s / 400 tokens, fresh code (LRU cache + tests) 3.4 s / 300 tokens.

## Memory

| card | used at utilization 0.93 (the served default) | at 0.95, after a 200k prefill |
|---|---|---|
| rank 0 (layers 0-9, embedding, Engram layer 1) | 58.0 GiB | 60.7 GiB |
| rank 1 (10-19, Engram layer 14, shadow of source 8) | 56.8 GiB | 59.5 GiB |
| rank 2 (20-29) | 57.5 GiB | 60.2 GiB |
| rank 3 (30-39, lm_head, shadows of 20 and 28) | 59.1 GiB | 61.8 GiB |

Cards on this box have crashed when driven to ~63 GiB, hence 0.93.

## Soak

`tools/soak.py 600 4` (agentic-style prompts, 1500 max tokens, temperature 1.0, four concurrent
streams, ten minutes) on the served default: 16 requests completed, 0 degenerate loops, no errors in
the server log, GPU memory flat at 58.0 / 56.8 / 57.5 / 59.1 GiB before and after (rounds of four
requests took ~155 s each, i.e. ~39 tok/s aggregate at concurrency 4).
