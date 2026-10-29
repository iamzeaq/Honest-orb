# Honest Orb brain speed results

Target: average under 3.5s on the evals, still 15/15.

Baseline measurement (3 questions, 1.3 + low reasoning): input about
11748 tokens per call, output 673-1086 tokens of which 397-747 were
reasoning tokens. Time-to-first-token was 5-19s, the dominant cost,
with only 2-3s of streaming tail.

| Change | Avg | Score |
|---|---|---|
| Baseline (1.3, low, uncapped) | 10.65s | 15/15 |
| Reasoning minimal | 4.97s | 15/15 |
| + cap 400 + brevity | 4.28s | 9/15, truncated, rejected |
| + cap 600 + brevity | 4.45s | 15/15 |
| Streaming instead of parse | no total gain | not kept |
| Caching | hits sometimes | prefix already static |
| 1.2 + minimal + cap 600 + brevity | 2.49s | 15/15, kept |
| Ablation, no cap or brevity | 3.10s | 15/15, cap+brevity help |
| Final, plus citation repair retry | 2.59s | 15/15 |

## Final config kept in api/brain.py

- Model: muse-spark-1.2-contributor
- Reasoning: minimal
- Max output tokens: 600
- Brevity: right, say_this and do_this at most 20 words each,
  spoken at most 3 short sentences
- Citation check against data/sections.json, plus one repair
  retry if an answered or emergency response has no citations
- All 35 sections still sent as the fixed system prefix
