"""Send four identical inputs; look for a near-one versus near-zero probability.

Requires only tinker==0.30.4 and TINKER_API_KEY. At most 20 forward calls,
10,240 input tokens, approximately $0.0076. No backward or optimizer calls.
"""

import asyncio
from datetime import datetime, timezone
import json
from itertools import combinations
import math
from pathlib import Path
from typing import Literal

import tinker

# Native GPT-OSS system header and user boundary; no tokenizer download needed.
HEADER = list(
    map(
        int,
        (
            "200006 17360 200008 3575 553 17554 162016 11 261 4410 6439 2359 22203 "
            "656 7788 17527 558 87447 100594 25 220 1323 19 12 3218 198 6576 3521 "
            "25 220 1323 21 12 3114 12 1130 279 30377 289 25 4465 279 2 13888 "
            "18403 25 8450 11 49159 11 1721 13 21030 2804 413 7360 395 1753 3176 "
            "13 200007 200006 1428 200008"
        ).split(),
    )
)
SENTENCE = [976, 9029, 328, 10128, 382, 12650, 558]  # The capital of France is Paris.\n
LENGTH = 128
BATCH_SIZE = 4
MAX_CALLS = 20
PROBABILITY_GAP_THRESHOLD = 0.99


def make_datum() -> tinker.Datum:
    """Shift one fixed token stream by one position, using all-one loss weights."""
    tokens = (HEADER + SENTENCE * LENGTH)[: LENGTH + 1]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=tokens[1:], dtype="int64", shape=[LENGTH]
            ),
            "weights": tinker.TensorData(
                data=[1.0] * LENGTH, dtype="float32", shape=[LENGTH]
            ),
        },
    )


def serialize(d: tinker.Datum) -> dict[str, object]:
    """Capture the complete token and tensor payload without credentials."""
    return {
        "input_tokens": d.model_input.to_ints(),
        "loss_fn_inputs": {
            k: {"data": v.data, "dtype": v.dtype, "shape": v.shape}
            for k, v in d.loss_fn_inputs.items()
        },
    }


def largest_probability_gap(
    logprobs: list[list[float]],
) -> tuple[float, int, int, int]:
    """Return the largest probability gap, its two row indices, and token position."""
    return max(
        (abs(math.exp(a) - math.exp(b)), i, j, position)
        for i, j in combinations(range(len(logprobs)), 2)
        for position, (a, b) in enumerate(zip(logprobs[i], logprobs[j], strict=True))
    )


async def main() -> None:
    """Save all scores and stop at a probability gap exceeding 99 percentage points."""
    output = (
        Path(__file__).resolve().parent
        / "artifacts"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-repro")
    )
    output.mkdir(parents=True)
    (output / "repro.py").write_bytes(Path(__file__).read_bytes())
    data = [make_datum() for _ in range(BATCH_SIZE)]
    payload = [serialize(d) for d in data]
    assert all(d == payload[0] for d in payload)
    (output / "request.json").write_text(json.dumps(payload))
    print(f"Evidence: {output}", flush=True)
    service = tinker.ServiceClient(max_retries=0)
    outcome: Literal["success", "errored", "interrupted"] = "errored"
    try:
        trainer = await service.create_lora_training_client_async(
            base_model="openai/gpt-oss-120b",
            rank=32,
            seed=0,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        (output / "trainer.json").write_text(
            json.dumps((await trainer.get_info_async()).model_dump(mode="json"))
        )
        for call in range(MAX_CALLS):
            result = await (
                await trainer.forward_async(data, loss_fn="cross_entropy")
            ).result_async()
            assert [serialize(d) for d in data] == payload
            assert len(result.loss_fn_outputs) == BATCH_SIZE
            logprobs = [
                [float(p) for p in d["logprobs"].data] for d in result.loss_fn_outputs
            ]
            assert all(len(row) == LENGTH for row in logprobs)
            assert all(math.isfinite(p) and p <= 0 for row in logprobs for p in row)
            gap, left_index, right_index, position = largest_probability_gap(logprobs)
            left, right = logprobs[left_index], logprobs[right_index]
            probabilities = [math.exp(left[position]), math.exp(right[position])]
            max_logprob_delta = max(
                abs(a - b)
                for i, j in combinations(range(BATCH_SIZE), 2)
                for a, b in zip(logprobs[i], logprobs[j], strict=True)
            )
            record = {
                "call": call,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "sdk_version": tinker.__version__,
                "metrics": result.metrics,
                "logprobs": logprobs,
                "max_abs_logprob_difference": max_logprob_delta,
                "max_abs_probability_difference": gap,
                "example_indices": [left_index, right_index],
                "position": position,
                "target_token": data[0].loss_fn_inputs["target_tokens"].data[position],
                "selected_logprob_difference": abs(left[position] - right[position]),
                "probabilities": probabilities,
                "reproduced": gap > PROBABILITY_GAP_THRESHOLD,
            }
            (output / f"result-{call:03d}.json").write_text(
                json.dumps(record, allow_nan=False)
            )
            print(
                f"call={call} examples={left_index},{right_index} position={position} "
                f"probabilities={probabilities[0]:.12g},{probabilities[1]:.12g} "
                f"max_abs_probability_difference={gap:.12g}",
                flush=True,
            )
            if gap > PROBABILITY_GAP_THRESHOLD:
                print(
                    "REPRODUCED: identical inputs in one request returned >99% versus <1% probability."
                )
                break
        else:
            print("INCONCLUSIVE: no probability gap above 99 percentage points in this bounded run.")
        outcome = "success"
    finally:
        await service.close(outcome)


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(main(), timeout=600))
