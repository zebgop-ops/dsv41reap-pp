# Four models on this box, one harness, identical inputs

Measured 2026-09-12. Each model served by its own stack on the four CMP 170HX cards, one at a
time. Perplexity per token is not comparable across tokenizers, so language modelling is
reported as **bits per byte**. MMLU uses the **same 1000 questions** for every model, cloze-scored
through the raw completions API (no chat template, so no thinking, 2-shot format anchor).
HumanEval+ / MBPP+ are pass@1 with the generated code executed against the EvalPlus tests in a
sandboxed container; anything that hit the first 4k token budget was retried at 16k so a verbose
reasoner is not scored down for thinking past the cap.

| model | wikitext bits/byte | code bits/byte | MMLU (1000q) | HumanEval+ | MBPP+ |
|---|---|---|---|---|---|
| Qwen3.8-Flash-Next FP8 | 0.4394 | 0.0805 | 89.0% | 94.5% (155/164) | 83.1% (314/378) |
| GLM-5.3-Flash W4A16 | 0.3884 | 0.1428 | 85.6% | 92.7% (152/164) | 83.3% (315/378) |
| DeepSeek-V4.1-Flash (unpruned) | 0.3275 | 0.1044 | 84.4% | 93.9% (154/164) | 84.9% (321/378) |
| DeepSeek-V4.1-Flash REAP-272E | 0.4091 | 0.1046 | 76.7% | 93.9% (154/164) | 83.3% (315/378) |
| *canonical ceiling* | | | | 99.4% (163/164) | 100% (378/378) |

## With speed (single stream, from each repo's results)

| model | decode | prefill | context |
|---|---|---|---|
| Qwen3.8-Flash-Next FP8 | 74 tok/s | ~10k tok/s | 262k |
| GLM-5.3-Flash W4A16 | 66-70 tok/s | ~3.2k tok/s | 512k |
| DeepSeek-V4.1-Flash (unpruned) | 14-22 tok/s | ~300 tok/s | 131k |
| DeepSeek-V4.1-Flash REAP-272E | ~30 tok/s | ~3k tok/s | 512k |

## Paired tests (row minus column, in points; * = McNemar p < 0.05)

### MMLU, same 1000 questions

| | Qwen3.8-Flash-Next | GLM-5.3-Flash | DeepSeek-V4.1-Flash | DeepSeek-V4.1-Flash |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** | - | +3.4* | +4.6* | +12.3* |
| **GLM-5.3-Flash** | -3.4* | - | +1.2 | +8.9* |
| **DeepSeek-V4.1-Flash** | -4.6* | -1.2 | - | +7.7* |
| **DeepSeek-V4.1-Flash** | -12.3* | -8.9* | -7.7* | - |

### HumanEval+ and MBPP+ combined, same tasks

| | Qwen3.8-Flash-Next | GLM-5.3-Flash | DeepSeek-V4.1-Flash | DeepSeek-V4.1-Flash |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** | - | +0.4 | -1.1 | +0.0 |
| **GLM-5.3-Flash** | -0.4 | - | -1.5 | -0.4 |
| **DeepSeek-V4.1-Flash** | +1.1 | +1.5 | - | +1.1 |
| **DeepSeek-V4.1-Flash** | +0.0 | +0.4 | -1.1 | - |

## Reading it

**No model is meaningfully better at code.** The four are spread over 2.0 points on the combined
542 tasks and no pair reaches significance in the paired test. The unpruned DeepSeek is nominally
top at 87.6%, but +1.1 over Qwen on 542 tasks is noise.

**REAP's pruning costs nothing on code and a lot on knowledge.** Pruned and unpruned tie exactly on
HumanEval+ (154/164 each) and sit 1.6 points apart on MBPP+, while the same checkpoints differ by
7.7 points on MMLU. Dropping 112 of 384 experts per layer took out knowledge, not coding ability.

**Verbosity varies wildly and does not buy accuracy.** GLM spends ~200 tokens per task, Qwen and
REAP ~1200. GLM finishes MBPP+ in 3.4 minutes against Qwen's 9.5 and scores 0.2 points higher. Five
to nine tasks per model still run past a 16k budget without closing a code block; those count as
failures, which is the right call since a user would see the same.

**Cross-checks.** The sandbox scores the datasets' own reference solutions at 378/378 (MBPP+) and
163/164 (HumanEval+, where one reference solution fails its own plus tests). Sampled failures are
genuine AssertionErrors, not harness artifacts.

## Caveats

- MMLU is cloze-scored with no chain of thought; code is generated with each model's default
  thinking behaviour. The two are not the same kind of measurement.
- 1000 MMLU questions and 542 code tasks: single-model sampling error is ±1.3 and ±1.5 points, so
  read the paired columns, not the margins between rows.
- Each stack has its own quantisation (Qwen FP8, GLM int4 experts, DeepSeek MXFP4 experts). This
  compares the servers as deployed, not the models at full precision.
- wikitext bits per byte favours DeepSeek, which carries 196 GB of Engram n-gram tables over a
  Wikipedia-derived corpus. Treat that column as partly a memorisation measure.
