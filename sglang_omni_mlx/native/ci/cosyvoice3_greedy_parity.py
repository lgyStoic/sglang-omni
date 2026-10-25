# SPDX-License-Identifier: Apache-2.0
"""Compare bounded independent native and Torch FP16 SDPA greedy trajectories."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2Model

from sglang_omni_mlx.native.ci.cosyvoice3_torch_parity import (
    checksum,
    metrics,
    require_validation_assertions,
    verified_weights,
)

# Note (lgyStoic): Pin separately encoded natural texts from CosyVoice-BlankEN, with endofprompt registered as 151646; no normalization/chat template.
PREFIX = "You are a helpful assistant.<|endofprompt|>"
PREFIX_IDS = [2610, 525, 264, 10950, 17847, 13, 151646]
TEXT_FIXTURES = [
    ("en_hello", "Hello, how are you?", [9707, 11, 1246, 525, 498, 30]),
    ("en_weather", "The sun is shining today.", [785, 7015, 374, 47925, 3351, 13]),
    (
        "zh_hello",
        "你好，很高兴见到你。",
        [56568, 52801, 3837, 99165, 44636, 99355, 88970, 26939, 56568, 1773],
    ),
    (
        "zh_weather",
        "今天天气很好，我们去散步吧。",
        [
            36171,
            35727,
            35727,
            99180,
            99165,
            52801,
            3837,
            35946,
            79478,
            85336,
            99632,
            64682,
            100003,
            1773,
        ],
    ),
]
SPEECH_TOKEN_COUNT = 6561
TASK_TOKEN_ID = 6563
VOCABULARY_SIZE = 6761


def main() -> int:
    require_validation_assertions()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-model", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "mps"], required=True)
    arguments = parser.parse_args()
    device = arguments.backend
    if device == "mps" and (
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
        or not torch.backends.mps.is_available()
    ):
        parser.error(
            "MPS requires an available device and PYTORCH_ENABLE_MPS_FALLBACK=0; no CPU fallback"
        )
    else:
        pass
    output = arguments.output
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report_path.write_text(json.dumps({"status": "incomplete"}))
    torch.set_num_threads(4)
    weights = verified_weights(arguments.raw_model, arguments.converted)
    config_path = arguments.raw_model / "CosyVoice-BlankEN/config.json"
    config = Qwen2Config.from_json_file(str(config_path))
    config._attn_implementation = "sdpa"
    model = Qwen2Model(config)
    backbone = {
        name.removeprefix("qwen2."): tensor
        for name, tensor in weights.items()
        if name.startswith("qwen2.")
    }
    model.load_state_dict(
        {name: tensor.to(device) for name, tensor in backbone.items()},
        strict=True,
        assign=True,
    )
    model.to(device).eval()
    for name, parameter in model.named_parameters():
        assert parameter.device.type == device and parameter.dtype == torch.float16
        assert torch.equal(parameter.cpu(), backbone[name])
    speech = weights["speech_embedding.weight"].to(device)
    head = weights["llm_decoder.weight"].to(device)
    assert torch.equal(speech.cpu(), weights["speech_embedding.weight"])
    assert torch.equal(head.cpu(), weights["llm_decoder.weight"])
    report = {
        "status": "incomplete",
        "backend": device,
        "dtype": "float16",
        "attention": "sdpa",
        "prefix": PREFIX,
        "prefix_ids": PREFIX_IDS,
        "tokenization": "pinned separately encoded prefix and target; tokenizer not executed",
        "weights_verified_exact": len(weights),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "mlx", "numpy"]
        },
        "platform": platform.platform(),
        "environment": {
            name: os.environ.get(name)
            for name in [
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
                "MLX_ENABLE_TF32",
            ]
        },
        "hashes": {
            str(path): checksum(path)
            for path in [
                Path(__file__),
                Path(__file__).with_name("cosyvoice3_torch_parity.py"),
                arguments.probe,
                arguments.raw_model / "llm.pt",
                config_path,
                arguments.converted / "model.safetensors",
                arguments.converted / "config.json",
            ]
        },
        "tolerance": {
            "atol": 0.02,
            "rtol": 0.002,
            "reference": "torch",
            "meaning": "diagnostic only",
        },
        "results": [],
    }
    with torch.inference_mode():
        for name, text, target_ids in TEXT_FIXTURES:
            text_ids = PREFIX_IDS + target_ids
            maximum_tokens = min(256, 20 * len(target_ids))
            minimum_tokens = min(2 * len(target_ids), maximum_tokens)
            native_logits_path = output / f"{name}.native.logits.npy"
            embeddings_path = output / f"{name}.native.embeddings.npy"
            command = [
                str(arguments.probe.resolve()),
                "--model-path",
                str(arguments.converted.resolve()),
                "--text-ids",
                json.dumps(text_ids),
                "--greedy-max-tokens",
                str(maximum_tokens),
                "--greedy-min-tokens",
                str(minimum_tokens),
                "--logits-output",
                str(native_logits_path.resolve()),
                "--embeddings-output",
                str(embeddings_path.resolve()),
            ]
            process = subprocess.run(
                command, capture_output=True, text=True, timeout=180
            )
            (output / f"{name}.process.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "returncode": process.returncode,
                        "stdout": process.stdout,
                        "stderr": process.stderr,
                    },
                    indent=2,
                )
            )
            process.check_returncode()
            native = json.loads(process.stdout)
            native_rows = np.load(native_logits_path)
            assert native_rows.shape == (len(native["ids"]), VOCABULARY_SIZE)
            assert (
                native["cache_token_count"]
                == 2 + len(text_ids) + len(native["ids"]) - 1
            )
            embeddings = torch.cat(
                [
                    speech[SPEECH_TOKEN_COUNT][None, None],
                    model.embed_tokens(torch.tensor([text_ids], device=device)),
                    speech[TASK_TOKEN_ID][None, None],
                ],
                dim=1,
            )
            assert np.array_equal(
                embeddings.float().cpu().numpy(), np.load(embeddings_path)
            )
            current = embeddings
            cache = None
            generated_ids = []
            rows = []
            stop_reason = "generation_cap"
            for step in range(maximum_tokens):
                prefix_length = embeddings.shape[1] + step
                result = model(
                    inputs_embeds=current,
                    attention_mask=torch.ones(
                        (1, prefix_length), dtype=torch.long, device=device
                    ),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = result.past_key_values
                assert cache.get_seq_length() == prefix_length
                assert all(
                    layer.keys.device.type == device
                    and layer.values.device.type == device
                    for layer in cache.layers
                )
                logits = torch.nn.functional.linear(
                    result.last_hidden_state[:, -1:, :], head
                )
                assert logits.device.type == device and logits.dtype == torch.float16
                scores = logits.float().cpu().numpy().reshape(VOCABULARY_SIZE)
                assert np.isfinite(scores).all()
                rows.append(scores)
                eligible = scores.copy()
                if step < minimum_tokens:
                    eligible[SPEECH_TOKEN_COUNT:] = -np.inf
                else:
                    pass
                selected_id = int(eligible.argmax())
                generated_ids.append(selected_id)
                if selected_id >= SPEECH_TOKEN_COUNT:
                    stop_reason = "control_token"
                    break
                else:
                    current = speech[selected_id][None, None]
            np.save(output / f"{name}.torch.logits.npy", np.stack(rows))
            divergence = next(
                (
                    step
                    for step, (left, right) in enumerate(
                        zip(native["ids"], generated_ids)
                    )
                    if left != right
                ),
                None,
            )
            if divergence is None and len(native["ids"]) != len(generated_ids):
                divergence = min(len(native["ids"]), len(generated_ids))
            else:
                pass
            common_rows = min(
                len(rows),
                len(native_rows),
                divergence + 1 if divergence is not None else len(rows),
            )
            row_metrics = [
                metrics(native_rows[step], rows[step]) for step in range(common_rows)
            ]
            comparison = {
                "name": name,
                "text": text,
                "target_ids": target_ids,
                "text_ids": text_ids,
                "min_tokens": minimum_tokens,
                "max_tokens": maximum_tokens,
                "native": native,
                "torch": {"ids": generated_ids, "stop_reason": stop_reason},
                "sequence_agreement": native["ids"] == generated_ids,
                "first_divergence_zero_based": divergence,
                "same_context_rows": row_metrics,
                "numerical_pass": all(row["violations"] == 0 for row in row_metrics),
            }
            report["results"].append(comparison)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            )
            print(
                f"{name}: selections={len(generated_ids)}, stop={stop_reason}, sequence_agreement={comparison['sequence_agreement']}",
                flush=True,
            )
    report["status"] = "completed"
    report["sequence_agreement"] = all(
        result["sequence_agreement"] for result in report["results"]
    )
    report["numerical_pass"] = all(
        result["numerical_pass"] for result in report["results"]
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"Completed; numerical_pass={report['numerical_pass']}, sequence_agreement={report['sequence_agreement']}"
    )
    return 0 if report["sequence_agreement"] and report["numerical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
