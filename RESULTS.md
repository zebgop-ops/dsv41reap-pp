# Results

DeepSeek-V4.1-Flash-REAP-272E on 4× CMP 170HX (sm_80, 64 GB, PCIe Gen2 x4), vLLM
`deepseekv41-flash-0909` + overlay, PP4 partition 10,10,10,10, every expert on the GPUs, n-gram
speculation (K=3, prompt-lookup 5..8), utilization 0.93, `--max-model-len 524288`. Single stream,
greedy, thinking off unless noted; "incl. prefill" numbers are wall time for the whole request.

## Correctness

- Greedy output of a 400-token code-edit task is byte-identical to the unpruned model's (and to the
  8,12,8,12 layout's, which only cuts on source boundaries).
- Needle-in-haystack recall OK at 30k and 200k tokens.

## What the pruning costs (measured, not the card's number)

Both checkpoints were served by this same stack and scored on byte-identical inputs with
`tools/quality.py` (perplexity from the server's `prompt_logprobs`; MMLU cloze-scored through the
raw completions API, so no chat template and no thinking, 2-shot format anchor). `tools/qcompare.py`
does the paired statistics; bits per byte is reported alongside perplexity because it is
tokenizer-independent and therefore comparable against other model families. The model card
reports +4.1% text perplexity; on a standard corpus the
loss is far larger, and it is concentrated in knowledge-heavy prose rather than code.

| metric | unpruned | REAP-272E | delta |
|---|---|---|---|
| wikitext-2 test perplexity (67,131 tokens, 36 windows) | 2.7147 | 3.4821 | **+28.3%** |
| — paired per-window NLL | | | +0.2495 nats ±0.0419 (95% CI) |
| Python stdlib perplexity (30,589 tokens, 16 windows) | 1.3633 | 1.3642 | +0.06% |
| wikitext-2 bits per byte (tokenizer-independent) | 0.3275 | 0.4091 | +24.9% |
| stdlib code bits per byte | 0.1044 | 0.1046 | +0.2% |
| — paired per-window NLL | | | +0.0004 nats ±0.0092 (not distinguishable from zero) |
| MMLU, 1000 questions, same questions both models | 84.4% | 76.7% | **−7.7 points** |
| HumanEval+ pass@1 (executed, same 164 tasks) | 93.9% (154) | 93.9% (154) | **0.0** |
| MBPP+ pass@1 (executed, same 378 tasks) | 84.9% (321) | 83.3% (315) | −1.6 (not significant) |
| — correct vs best distractor, mean margin | 3.40 nats | 2.14 nats | −1.26 nats |

Paired MMLU: 730 both correct, 114 the unpruned model alone, 37 REAP alone, 119 both wrong; the two
models pick the same letter on 81.7% of questions. McNemar on the 151 discordant pairs gives
p < 0.0001, so the gap is not sampling noise. Worst subjects (≥12 questions): high-school chemistry
−33 pts, professional law −21.5, prehistory −21, professional accounting and security studies −19.
A few move the other way (high-school physics +17, geography +8), consistent with reshuffling rather
than uniform damage.

**This is the checkpoint, not the port.** `ktests/test_reap_router.py` checks vLLM's Triton
`dsv4_topk` (which `patch_reap_router.py` lets accept 272 experts) against a torch transcription of
the checkpoint's own `Gate.forward` for expert counts 128/160/256/272/384 at 1..2048 tokens: expert
selection identical, weights within 1e-5, 16/16 pass. Independently, code perplexity is unchanged
and a 400-token greedy code edit is byte-identical to the unpruned model's output — a mis-routing
bug could not leave those intact while costing 8 points of MMLU.

**Reading it.** Code is untouched, and that now rests on executed tests rather than perplexity
alone: HumanEval+ ties exactly and MBPP+ is within noise. Factual and reasoning-heavy prose loses
real ground. If you serve this checkpoint for coding, the 2-3x speedup is close to free. If you
serve it for knowledge questions, budget for roughly the accuracy of a substantially smaller model.

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
