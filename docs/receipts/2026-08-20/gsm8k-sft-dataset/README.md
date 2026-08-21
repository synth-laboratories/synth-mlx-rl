# gsm8k_sft_dataset.py — receipt (2026-08-20)

One real run of `scripts/gsm8k_sft_dataset.py` against the merged containers
pin (containers `9916cd7479c1029d7dd9091db43f52af1cdf484e`), offline
(`HF_HUB_OFFLINE=1`, pinned revision already in the HF cache):

```
gsm8k_sft_dataset.py --out <dir> --train 64 --eval 16 --containers-src <containers>/src
64 train rows  -> gsm8k-train-64-a0558fa4ec6c.jsonl
16 heldout rows -> gsm8k-eval-16-4ffaa737ff28.jsonl
openai/gsm8k @ 740312add88f  train sha256:dca449882e67..  heldout sha256:32c548f08195..  shuffle_seed=20260820
```

`manifest.json` here is the run's manifest verbatim: the dataset pin (revision,
split digests, shuffle seed, declared profile), the seeds each file covers, and
each file's own sha256 (which is also in its filename). The JSONL files are not
committed; they are reproducible from the pin.
