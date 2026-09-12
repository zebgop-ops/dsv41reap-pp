# Four models on this box, same harness, same inputs

Measured 2026-09-12. Each model served by its own stack on the four CMP 170HX cards (one at a
time), scored with `tools/quality.py`: perplexity from the server's `prompt_logprobs`, MMLU
cloze-scored through the raw completions API (no chat template, so no thinking, 2-shot format
anchor), the **same 1000 MMLU questions** for every model.

Perplexity per token cannot be compared across tokenizers, so the table ranks on **bits per byte**,
which can. MMLU accuracy is tokenizer-independent.

| model | wikitext-2 bits/byte | code bits/byte | MMLU (1000 q) | confidence margin |
|---|---|---|---|---|
| Qwen3.8-Flash-Next FP8 | 0.4394 | **0.0805** | **89.0%** | 3.91 |
| GLM-5.3-Flash W4A16 | 0.3884 | 0.1428 | 85.6% | **3.94** |
| DeepSeek-V4.1-Flash (unpruned) | **0.3275** | 0.1044 | 84.4% | 3.40 |
| DeepSeek-V4.1-Flash REAP-272E | 0.4091 | 0.1046 | 76.7% | 2.14 |

Lower bits per byte is better. Margin = mean logprob gap between the correct option and the best
distractor, in nats.

## Paired MMLU (row minus column, in points; * = McNemar p < 0.05)

|  | DSv41 | REAP | Qwen3.8 | GLM-5.3 |
|---|---|---|---|---|
| **DSv41 unpruned** | – | +7.7* | −4.6* | −1.2 |
| **DSv41R REAP** | −7.7* | – | −12.3* | −8.9* |
| **Qwen3.8** | +4.6* | +12.3* | – | +3.4* |
| **GLM-5.3** | +1.2 | +8.9* | −3.4* | – |

Only DSv41 vs GLM-5.3 is a statistical tie. Everything else separates.

## With speed (single stream, from each repo's results)

| model | decode | prefill | context | MMLU |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | 74 tok/s | ~10k tok/s | 262k | 89.0% |
| GLM-5.3-Flash | 66–70 tok/s | ~3.2k tok/s | 512k | 85.6% |
| DSv41R REAP | ~30 tok/s | ~3k tok/s | 512k | 76.7% |
| DSv41 unpruned | 14–22 tok/s | ~300 tok/s | 131k | 84.4% |

**Qwen3.8 is both the fastest and the most accurate**, and it models code best. On this hardware it
is the default choice for almost everything.

The DeepSeek pair only leads on one axis: wikitext bits per byte, where the unpruned model is
clearly best. Read that with care — DeepSeek-V4.1 carries 196 GB of Engram n-gram tables, an
explicit memorisation device, and wikitext is Wikipedia. That is plausibly recall of the corpus
rather than better generalisation, which is consistent with it placing third on MMLU.

REAP-272E is last on MMLU by 8 to 12 points and is slower than both Qwen and GLM. Its case is
narrow: DeepSeek-family behaviour with a 512k context at 2–3x the unpruned speed, for code-shaped
work where its perplexity is unaffected.

## Caveats

- Cloze scoring with no chain of thought. All four models were handicapped identically, so the
  comparison is fair, but the absolute numbers are lower than these models' CoT benchmark figures.
- 1000 MMLU questions: ±1.3 points of sampling error on a single model, much tighter on the paired
  differences above, which is why the paired table is the one to read.
- Each stack has its own quantisation (Qwen FP8, GLM int4 experts, DeepSeek MXFP4 experts), so this
  compares *the servers as deployed*, not the models at full precision.
