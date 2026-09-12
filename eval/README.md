# Quality evaluation fixtures

`tools/quality.py` needs two corpora; neither is committed (a few MB of parquet). Regenerate:

```bash
hf download Salesforce/wikitext --repo-type dataset --include "wikitext-2-raw-v1/test*" --local-dir eval/wikitext
hf download cais/mmlu --repo-type dataset --include "all/test-*" --local-dir eval/mmlu
python3 -m venv eval/.venv && eval/.venv/bin/pip install pyarrow    # the image has no pyarrow
eval/.venv/bin/python - <<'PY'
import pyarrow.parquet as pq, json
t = pq.read_table('eval/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet').to_pydict()
open('eval/wikitext2-test.txt','w').write(''.join(t['text']))
m = pq.read_table('eval/mmlu/all/test-00000-of-00001.parquet').to_pydict()
json.dump([{'q':m['question'][i],'choices':list(m['choices'][i]),'answer':int(m['answer'][i]),
            'subject':m['subject'][i]} for i in range(len(m['question']))], open('eval/mmlu-test.json','w'))
PY
```

`eval/stdlib-code.txt` is six Python standard-library modules concatenated (dataclasses, argparse,
json/decoder, http/client, asyncio/tasks, statistics) as a code-domain corpus.

Run against a server, then compare:

```bash
DSV41_MODEL=DSv41Flash     python3 tools/quality.py ppl http://localhost:8004 eval/wikitext2-test.txt 73728 8192
DSV41_MODEL=DSv41Flash DSV41_MC_DUMP=eval/mc-base-1000.json python3 tools/quality.py mc http://localhost:8004 eval/mmlu-test.json 1000 0
# ... same on the other checkpoint, then
python3 tools/qcompare.py eval/base.txt eval/reap.txt eval/mc-base-1000.json eval/mc-reap-1000.json
```
