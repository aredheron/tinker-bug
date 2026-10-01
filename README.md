# Tinker duplicate log-probability repro

`repro.py` creates a fresh `openai/gpt-oss-120b` LoRA client (rank 32,
seed 0) and submits **four identical examples in one `forward` call** with
`loss_fn="cross_entropy"`.

Each example contains 128 input tokens: a native GPT-OSS system/user header
followed by repetitions of `The capital of France is Paris.\n`. Targets are
shifted one token ahead, with all loss weights set to one. Tinker returns
128 target-token log-probabilities per example. No sampling, backward passes,
or optimizer updates occur.

## Run

Copy `.env.example` to `.env`, fill in `TINKER_API_KEY`, then run:

```bash
uv run --env-file .env python repro.py
```

The script checks all six pairs of examples and makes at most 20 forward calls,
stopping when the same target token's probabilities differ by more than
99 percentage points (>99% versus <1%). Inputs and results are saved in `artifacts/`.

## Observed result

The updated script reproduced this on its third call at zero-based position 122,
scoring ` France` after the identical prefix ending in `The capital of`:

| | Second example | Third example |
|---|---:|---:|
| Log-probability | −0.000144709600 | −15.599808693 |
| Probability | **99.9855301%** | **0.0000167915%** |

The probability gap is **99.9855133 percentage points**; the log-probability
difference is **15.599664 natural-log units (nats)**. The first two calls returned
matching scores, so the discrepancy does not occur on every call.

Is this expected behavior, or a backend correctness issue?
