# DeepSeek-V4.1-Flash-REAP-272E on 4× sm_80: every expert on the GPUs

Serving [LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E)
(DeepSeek-V4.1-Flash with 272 of its 384 routed experts per layer kept by REAP pruning, native
MXFP4 experts / FP8 dense / FP8 Engram) on four CMP 170HX cards (Ampere sm_80, 64 GB, PCIe Gen2 x4,
no P2P) with vLLM's official `deepseekv41-flash-0909` image plus a Python overlay. Sibling of
[dsv41flash-pp](https://github.com/zebgop-ops/dsv41flash-pp) (the unpruned model on the same box,
which needs nine layers' experts on the CPU), [glm53flash-pp](https://github.com/zebgop-ops/glm53flash-pp)
and [qwen38-flashnext-pp](https://github.com/zebgop-ops/qwen38-flashnext-pp).

The model card says the checkpoint "cannot run on vLLM" because 272 experts fall outside a kernel's
expert-count table. It does run; what it takes is in [FINDINGS.md](FINDINGS.md), the numbers in
[RESULTS.md](RESULTS.md).

## What the pruning buys on this hardware

Experts drop from 6.72 to 4.76 GiB per layer, so a 10,10,10,10 pipeline partition holds all 40
layers' experts on the cards (~52 GiB per rank). The unpruned model has to stream nine layers'
experts from host RAM on every step, which capped decode at 14-22 tok/s and prefill at ~100 tok/s.

| | unpruned (9 CPU layers) | REAP-272E, all GPU |
|---|---|---|
| free prose decode | ~14 tok/s | ~30 tok/s |
| code (copy-heavy edits / fresh code) | ~22 / ~20 tok/s | ~60-110 / ~85 tok/s |
| prefill, 14k-token prompt | ~100 tok/s | ~3k tok/s |
| KV pool at the served context | 0.9M tokens @131k | see RESULTS.md |
| MMLU (1000 questions, cloze-scored) | 84.4% | 76.7% |
| wikitext-2 perplexity | 2.715 | 3.482 |

Cost, measured on this box with both checkpoints on the same stack and identical inputs
([RESULTS.md](RESULTS.md)): **wikitext-2 perplexity +28.3%** and **MMLU −7.7 points** (84.4% → 76.7%
over the same 1000 questions, McNemar p < 0.0001), against the +4.1% text perplexity the model card
reports. Code is the exception: stdlib perplexity moves +0.06% (inside the noise) and a 400-token
greedy code edit is byte-identical to the unpruned model's output. The port itself is exonerated by
`ktests/test_reap_router.py`, which matches the checkpoint's reference gate at 272 experts.
So: near-free for coding, a real knowledge hit for everything else.

## How it compares to the other models on this box

Same harness, same 1000 MMLU questions, each model on its own stack ([CROSS-MODEL.md](CROSS-MODEL.md)):

| model | wikitext bits/byte | code bits/byte | MMLU | decode |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | 0.4394 | 0.0805 | 89.0% | 74 tok/s |
| GLM-5.3-Flash W4A16 | 0.3884 | 0.1428 | 85.6% | 66-70 tok/s |
| DeepSeek-V4.1-Flash unpruned | 0.3275 | 0.1044 | 84.4% | 14-22 tok/s |
| this checkpoint (REAP-272E) | 0.4091 | 0.1046 | 76.7% | ~30 tok/s |

Qwen3.8 is both faster and more accurate than everything else here, so reach for this checkpoint
when you specifically want DeepSeek-family behaviour with a 512k context on code-shaped work.

## Quickstart

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909
hf download deepseek-ai/DeepSeek-V4.1-Flash            # needed once for its two Engram shards (189 GiB)
hf download LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E   # 208 GiB
./serve/link-reap-engram.sh                             # links shards 47/48 into the REAP snapshot (relative links)
docker run --rm -v $PWD/overlay/engram_ssd:/w --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 /w/build.sh
docker run --rm -v $PWD/overlay/hybrid:/w --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 /w/build.sh
DSV41_HF=$HOME/.cache/huggingface ./serve/run-dsv41reap-pp4.sh   # DSv41ReapFlash on :8005
```

`serve/run-dsv41reap-pp4.sh` wraps the generic launcher (`serve/run-dsv41-pp4.sh`, shared with
dsv41flash-pp) with this checkpoint's defaults: partition `10,10,10,10`, no CPU expert layers,
n-gram (prompt-lookup) speculation, GPU utilization 0.95 (every card keeps ~5 GiB of slack), and
`--max-model-len 524288`. The model's own DSpark drafter (`DSV41_SPEC_METHOD=dspark`) is faster on
fresh code but its 8.4 GiB draft only fits on the last rank if that rank's last two layers keep
their experts on the CPU, so it is not the default. Other knobs as in dsv41flash-pp:
`DSV41_PARTITION`, `DSV41_MAXLEN`, `DSV41_UTIL`, `DSV41_SEQS`, `DSV41_BATCHED` (prefill chunk),
`DSV41_DSPARK_CONF` (confidence truncation of DSpark drafts), and the diagnostic switches in
FINDINGS.md.

The launcher needs the `hybrid` native CPU MoE built (`overlay/hybrid/build.sh`) even when no layer
uses it, and `kt/site` (kt-kernel) only for the legacy `DSV41_CPU_MOE=kt` backend.

## Layout

```
serve/run-dsv41reap-pp4.sh  this checkpoint's launcher (wraps run-dsv41-pp4.sh)
serve/link-reap-engram.sh   links the base model's Engram shards into the REAP snapshot
overlay/vllm/               the Python overlay mounted over the image's vllm package
overlay/patch_*.py          anchor-based, idempotent patch scripts (apply-overlay.sh runs them all);
                            REAP-specific: patch_reap_router.py, and the PP shadow plan (pp_shadow.py)
                            that makes cuts inside an index group legal
overlay/hybrid/             pp_shadow.py, cpu_experts.py + cpu_moe.cpp, dspark_proposer.py (DSpark on the V1 runner)
overlay/engram_ssd/         the Engram row store (tables read from the NVMe shards)
patches/vllm-overlay.diff   full delta vs the pristine image
tools/                      prefill.py, itl.py, specbench.py, needle.py, soak.py, lpcheck.py, tracetl.py ...
```

## Credits

Same lineage as dsv41flash-pp: vLLM's DeepSeek-V4.1 port, haosdent's sm_80 work for V4, 0xSero's
Engram row-store design, and LibertAIDAI for the REAP checkpoint.
